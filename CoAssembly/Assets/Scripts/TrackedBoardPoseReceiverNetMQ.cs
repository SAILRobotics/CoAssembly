using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using Newtonsoft.Json;

public class TrackedBoardPoseReceiverNetMQ : MonoBehaviour
{
    [Header("Target")]
    public Transform boardRoot;

    [Header("NetMQ")]
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int port = 5014;

    [Header("Behavior")]
    [SerializeField] private bool applyContinuously = true;

    [Header("Manual Alignment Offset")]
    [Tooltip("Fine-tune position (metres) if BoardRoot appears misaligned from the real ArUco marker.")]
    [SerializeField] private Vector3 positionOffset = Vector3.zero;
    [Tooltip("Fine-tune rotation (degrees) applied after the received pose.")]
    [SerializeField] private Vector3 rotationOffset = Vector3.zero;

    private Thread receiveThread;
    private volatile bool isRunning = false;
    private bool hasShutdown = false;
    private SubscriberSocket subscriber;

    private readonly ConcurrentQueue<float[]> matrixQueue = new();

    [Serializable]
    public class BoardRootMatrixData
    {
        public float[] board_root_matrix;
    }

void Start()
{
    Debug.Log("[TrackedBoardPoseReceiver] 🟢 (1) Start() called");

    if (boardRoot == null)
    {
        Debug.LogError("[TrackedBoardPoseReceiver] ❌ boardRoot is not assigned.");
        enabled = false;
        return;
    }

    Debug.Log("[TrackedBoardPoseReceiver] 🟢 (2) boardRoot OK");

    NetMQManager.RegisterReceiver();

    Debug.Log("[TrackedBoardPoseReceiver] 🟢 (3) Registered with NetMQManager");

    isRunning = true;
    receiveThread = new Thread(ReceiveLoop)
    {
        IsBackground = true
    };
    receiveThread.Start();

    Debug.Log($"[TrackedBoardPoseReceiver] 📡 (4) Listening on tcp://0.0.0.0:{port}");
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

                Debug.Log($"[TrackedBoardPoseReceiver] 🔗 Bound to {address}");

                while (isRunning)
                {
                    try
                    {
                        if (subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string message))
                        {
                            var data = JsonConvert.DeserializeObject<BoardRootMatrixData>(message);
                            if (data?.board_root_matrix != null && data.board_root_matrix.Length == 16)
                            {
                                matrixQueue.Enqueue(data.board_root_matrix);
                            }
                            else
                            {
                                Debug.LogWarning("[TrackedBoardPoseReceiver] ⚠️ Invalid board_root_matrix received.");
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
                            Debug.LogWarning("[TrackedBoardPoseReceiver] ❌ Error: " + e.Message);
                    }
                }
            }
        }
        catch (Exception e)
        {
            if (isRunning)
                Debug.LogWarning("[TrackedBoardPoseReceiver] ❌ ReceiveLoop outer exception: " + e.Message);
        }
    }

    void Update()
    {
        while (matrixQueue.TryDequeue(out var flat16))
        {
            ApplyMatrixToBoardRoot(flat16);

            if (!applyContinuously)
                break;
        }

        if (NetMQManager.IsShutdownRequested)
            ShutdownNetMQ();
    }

    private void ApplyMatrixToBoardRoot(float[] flat16)
    {
        if (flat16 == null || flat16.Length != 16 || boardRoot == null)
        {
            Debug.LogWarning("[TrackedBoardPoseReceiver] ⚠️ Invalid matrix or missing boardRoot.");
            return;
        }

        Matrix4x4 mat = new Matrix4x4();
        for (int i = 0; i < 16; i++)
            mat[i] = flat16[i];   // Unity Matrix4x4 indexer is column-major

        if (!IsValidMatrix(mat))
        {
            Debug.LogWarning("[TrackedBoardPoseReceiver] ⚠️ Non-homogeneous matrix; skipping.");
            return;
        }

        Vector3 pos = mat.GetColumn(3);
        Quaternion rot = mat.rotation;

        // Apply manual offset: position is shifted along the received rotation axes,
        // then rotation offset is added on top.
        pos += rot * positionOffset;
        rot *= Quaternion.Euler(rotationOffset);

        boardRoot.localPosition = pos;
        boardRoot.localRotation = rot;

        Debug.Log($"[TrackedBoardPoseReceiver] ✅ Applied BoardRoot pose: pos={pos}, rot={rot.eulerAngles}");
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

        Debug.Log("[TrackedBoardPoseReceiver] 🔻 Shutting down...");

        try
        {
            isRunning = false;

            subscriber?.Close();
            subscriber?.Dispose();
            subscriber = null;

            if (receiveThread != null && receiveThread.IsAlive)
            {
                if (!receiveThread.Join(1000))
                    Debug.LogWarning("[TrackedBoardPoseReceiver] ⚠️ Receive thread did not exit within timeout.");
            }

            NetMQManager.UnregisterReceiver();
            Debug.Log("[TrackedBoardPoseReceiver] ✅ Shutdown complete");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[TrackedBoardPoseReceiver] ⚠️ Shutdown exception: " + e.Message);
        }
    }

    private void OnDestroy() => ShutdownNetMQ();
    private void OnApplicationQuit() => ShutdownNetMQ();
}
