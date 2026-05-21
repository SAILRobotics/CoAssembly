using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using Newtonsoft.Json;

public class PegboardRootPoseReceiverNetMQ : MonoBehaviour
{
    [Header("Target")]
    public Transform pegboardRoot;

    [Header("NetMQ")]
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int port = 5008;

    [Header("Behavior")]
    [SerializeField] private bool applyContinuously = true;

    [Header("Manual Alignment Offset")]
    [Tooltip("Fine-tune position (metres) if PegboardRoot appears misaligned from the real ArUco marker.")]
    [SerializeField] private Vector3 positionOffset = Vector3.zero;
    [Tooltip("Fine-tune rotation (degrees) applied after the received pose.")]
    [SerializeField] private Vector3 rotationOffset = Vector3.zero;

    private Thread receiveThread;
    private volatile bool isRunning = false;
    private bool hasShutdown = false;
    private SubscriberSocket subscriber;

    private readonly ConcurrentQueue<float[]> matrixQueue = new();

    [Serializable]
    public class PegboardRootMatrixData
    {
        public float[] pegboard_root_matrix;
    }

void Start()
{
    Debug.Log("[PegboardRootPoseReceiver] 🟢 (1) Start() called");

    if (pegboardRoot == null)
    {
        Debug.LogError("[PegboardRootPoseReceiver] ❌ pegboardRoot is not assigned.");
        enabled = false;
        return;
    }

    Debug.Log("[PegboardRootPoseReceiver] 🟢 (2) pegboardRoot OK");

    NetMQManager.RegisterReceiver();

    Debug.Log("[PegboardRootPoseReceiver] 🟢 (3) Registered with NetMQManager");

    isRunning = true;
    receiveThread = new Thread(ReceiveLoop)
    {
        IsBackground = true
    };
    receiveThread.Start();

    Debug.Log($"[PegboardRootPoseReceiver] 📡 (4) Listening on tcp://0.0.0.0:{port}");
}

    private void ReceiveLoop()
    {
        AsyncIO.ForceDotNet.Force();

        try
        {
            using (subscriber = new SubscriberSocket())
            {
                string address = $"tcp://0.0.0.0:{port}";
                subscriber.Bind(address);
                subscriber.Subscribe("");

                Debug.Log($"[WorldRootPoseReceiver] 🔗 Bound to {address}");

                while (isRunning)
                {
                    try
                    {
                        if (subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string message))
                        {
                            var data = JsonConvert.DeserializeObject<PegboardRootMatrixData>(message);
                            if (data?.pegboard_root_matrix != null && data.pegboard_root_matrix.Length == 16)
                            {
                                matrixQueue.Enqueue(data.pegboard_root_matrix);
                            }
                            else
                            {
                                Debug.LogWarning("[PegboardRootPoseReceiver] ⚠️ Invalid pegboard_root_matrix received.");
                            }
                        }
                    }
                    catch (TerminatingException)
                    {
                        break;
                    }
                    catch (ObjectDisposedException)
                    {
                        break;
                    }
                    catch (Exception e)
                    {
                        if (isRunning)
                            Debug.LogWarning("[PegboardRootPoseReceiver] ❌ Error: " + e.Message);
                    }
                }
            }
        }
        catch (Exception e)
        {
            if (isRunning)
                Debug.LogWarning("[PegboardRootPoseReceiver] ❌ ReceiveLoop outer exception: " + e.Message);
        }
    }

    void Update()
    {
        while (matrixQueue.TryDequeue(out var flat16))
        {
            ApplyMatrixToPegboardRoot(flat16);

            if (!applyContinuously)
                break;
        }

        if (NetMQManager.IsShutdownRequested)
            ShutdownNetMQ();
    }

    private void ApplyMatrixToPegboardRoot(float[] flat16)
    {
        if (flat16 == null || flat16.Length != 16 || pegboardRoot == null)
        {
            Debug.LogWarning("[PegboardRootPoseReceiver] ⚠️ Invalid matrix or missing pegboardRoot.");
            return;
        }

        Matrix4x4 mat = new Matrix4x4();
        for (int i = 0; i < 16; i++)
            mat[i] = flat16[i];   // Unity Matrix4x4 indexer is column-major

        if (!IsValidMatrix(mat))
        {
            Debug.LogWarning("[PegboardRootPoseReceiver] ⚠️ Non-homogeneous matrix; skipping.");
            return;
        }

        Vector3 pos = mat.GetColumn(3);
        Quaternion rot = mat.rotation;

        // Apply manual offset: position is shifted along the received rotation axes,
        // then rotation offset is added on top.
        pos += rot * positionOffset;
        rot *= Quaternion.Euler(rotationOffset);

        pegboardRoot.localPosition = pos;
        pegboardRoot.localRotation = rot;

        Debug.Log($"[PegboardRootPoseReceiver] ✅ Applied PegboardRoot pose: pos={pos}, rot={rot.eulerAngles}");
    }

    private bool IsValidMatrix(Matrix4x4 mat)
    {
        Vector4 row3 = mat.GetRow(3);
        return Mathf.Approximately(row3.x, 0f) &&
               Mathf.Approximately(row3.y, 0f) &&
               Mathf.Approximately(row3.z, 0f) &&
               Mathf.Approximately(row3.w, 1f);
    }

    public void ShutdownNetMQ()
    {
        if (hasShutdown) return;
        hasShutdown = true;

        Debug.Log("[PegboardRootPoseReceiver] 🔻 Shutting down...");

        try
        {
            isRunning = false;

            subscriber?.Close();
            subscriber?.Dispose();
            subscriber = null;

            if (receiveThread != null && receiveThread.IsAlive)
            {
                if (!receiveThread.Join(1000))
                    Debug.LogWarning("[WorldRootPoseReceiver] ⚠️ Receive thread did not exit within timeout.");
            }

            NetMQManager.UnregisterReceiver();
            Debug.Log("[WorldRootPoseReceiver] ✅ Shutdown complete");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[WorldRootPoseReceiver] ⚠️ Shutdown exception: " + e.Message);
        }
    }

    private void OnDestroy() => ShutdownNetMQ();
    private void OnApplicationQuit() => ShutdownNetMQ();
}
