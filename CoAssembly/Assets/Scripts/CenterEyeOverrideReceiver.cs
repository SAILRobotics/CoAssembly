using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using Newtonsoft.Json;

/// <summary>
/// Receives a pose override for CenterEyeAnchor from Python and applies it
/// every LateUpdate() — after OVR has done its own tracking update — so the
/// override wins even if rotation tracking is still enabled.
///
/// Attach to any persistent GameObject.  Drag OVRCameraRig's CenterEyeAnchor
/// into the centerEyeAnchor field in the Inspector.
///
/// Requires OVRManager.usePositionTracking = false (set in the Inspector on
/// OVRManager, or at runtime) so OVR does not overwrite the position each frame.
/// </summary>
public class CenterEyeOverrideReceiver : MonoBehaviour
{
    [Header("Target")]
    public Transform centerEyeAnchor;

    [Header("NetMQ")]
    [SerializeField] private int port = 5016;

    [Header("Debug")]
    [SerializeField] private bool verboseLogs = false;

    private Thread receiveThread;
    private volatile bool isRunning = false;
    private bool hasShutdown = false;
    private SubscriberSocket subscriber;

    private readonly ConcurrentQueue<float[]> matrixQueue = new();

    [Serializable]
    private class PoseData
    {
        public float[] center_eye_matrix;
    }

    void Start()
    {
        if (centerEyeAnchor == null)
        {
            Debug.LogError("[CenterEyeOverride] centerEyeAnchor is not assigned.");
            enabled = false;
            return;
        }

        NetMQManager.RegisterReceiver();

        isRunning = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();

        Debug.Log($"[CenterEyeOverride] Listening on tcp://0.0.0.0:{port}");
    }

    private void ReceiveLoop()
    {
        AsyncIO.ForceDotNet.Force();
        try
        {
            using (subscriber = new SubscriberSocket())
            {
                subscriber.Bind($"tcp://0.0.0.0:{port}");
                subscriber.Subscribe("");

                while (isRunning)
                {
                    try
                    {
                        if (subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string msg))
                        {
                            var data = JsonConvert.DeserializeObject<PoseData>(msg);
                            if (data?.center_eye_matrix != null && data.center_eye_matrix.Length == 16)
                                matrixQueue.Enqueue(data.center_eye_matrix);
                        }
                    }
                    catch (TerminatingException) { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e)
                    {
                        if (isRunning)
                            Debug.LogWarning("[CenterEyeOverride] ReceiveLoop error: " + e.Message);
                    }
                }
            }
        }
        catch (Exception e)
        {
            if (isRunning)
                Debug.LogWarning("[CenterEyeOverride] ReceiveLoop outer: " + e.Message);
        }
    }

    void LateUpdate()
    {
        // Drain all but the latest message.
        float[] latest = null;
        while (matrixQueue.TryDequeue(out var flat16))
            latest = flat16;

        if (latest != null)
            ApplyMatrix(latest);

        if (NetMQManager.IsShutdownRequested)
            Shutdown();
    }

    private void ApplyMatrix(float[] flat16)
    {
        Matrix4x4 mat = new Matrix4x4();
        for (int i = 0; i < 16; i++)
            mat[i] = flat16[i];

        Vector4 row3 = mat.GetRow(3);
        bool valid = Mathf.Approximately(row3.x, 0f) &&
                     Mathf.Approximately(row3.y, 0f) &&
                     Mathf.Approximately(row3.z, 0f) &&
                     Mathf.Approximately(row3.w, 1f);
        if (!valid) return;

        Vector3 pos = mat.GetColumn(3);
        Quaternion rot = mat.rotation;

        centerEyeAnchor.SetPositionAndRotation(pos, rot);

        if (verboseLogs)
            Debug.Log($"[CenterEyeOverride] pos={pos} rot={rot.eulerAngles}");
    }

    public void Shutdown()
    {
        if (hasShutdown) return;
        hasShutdown = true;

        isRunning = false;
        subscriber?.Close();
        subscriber?.Dispose();
        subscriber = null;

        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Join(1000);

        NetMQManager.UnregisterReceiver();
        Debug.Log("[CenterEyeOverride] Shutdown complete");
    }

    private void OnDestroy() => Shutdown();
    private void OnApplicationQuit() => Shutdown();
}
