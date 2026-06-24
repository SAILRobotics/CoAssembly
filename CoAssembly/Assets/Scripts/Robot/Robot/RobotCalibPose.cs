using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using Newtonsoft.Json;
using System.Linq;

public class RobotBaseInitialPoseReceiverNetMQ : MonoBehaviour
{
    [Header("Targets")]
    public Transform robotBase;
    [Tooltip("The WorldRoot GameObject whose transform represents the calibrated world frame. " +
             "Robot pose is sent in that frame and must be converted to Unity world space here.")]
    public Transform worldRoot;

    [Header("Behavior")]
    [Tooltip("If true, only the first received matrix is applied and all later ones are ignored. " +
             "Disable to re-apply every time Python sends a new matrix (useful for tuning).")]
    [SerializeField] private bool applyOnce = false;

    [Header("NetMQ")]
    [SerializeField] private int port = 5000;

    // Threading / NetMQ
    private Thread receiveThread;
    private bool isRunning = false;
    private bool hasShutdown = false;
    private SubscriberSocket subscriber;

    // Queue of incoming matrices
    private readonly ConcurrentQueue<float[]> robotMatrixQueue = new();

    private bool robotPoseApplied = false;

    void Start()
    {
        if (robotBase == null)
        {
            Debug.LogError("[RobotBaseInitialPoseReceiver] ❌ robotBase is not assigned.");
            enabled = false;
            return;
        }

        if (worldRoot == null)
            Debug.LogWarning("[RobotBaseInitialPoseReceiver] ⚠️ worldRoot not assigned — " +
                             "robot pose will be treated as Unity world-space (no WorldRoot transform applied).");

        isRunning = true;
        receiveThread = new Thread(ReceiveLoop)
        {
            IsBackground = true
        };
        receiveThread.Start();

        NetMQManager.RegisterReceiver();
        Debug.Log($"[RobotBaseInitialPoseReceiver] 📡 Bound on tcp://0.0.0.0:{port}");
    }

    private void ReceiveLoop()
    {
        AsyncIO.ForceDotNet.Force();

        using (subscriber = new SubscriberSocket())
        {
            subscriber.Bind($"tcp://0.0.0.0:{port}");
            subscriber.Subscribe("");

            while (isRunning)
            {
                try
                {
                    if (subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string message))
                    {
                        var data = JsonConvert.DeserializeObject<RobotMatrixData>(message);
                        if (data?.robot_matrix != null && data.robot_matrix.Length == 16)
                        {
                            robotMatrixQueue.Enqueue(data.robot_matrix);
                            // 🔹 Optional: lightweight log that we actually got something
                            Debug.Log("[RobotBaseInitialPoseReceiver] 📥 Enqueued robot_matrix from ZMQ.");
                        }
                        else
                        {
                            Debug.LogWarning("[RobotBaseInitialPoseReceiver] ⚠️ Invalid robot_matrix received.");
                        }
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[RobotBaseInitialPoseReceiver] ❌ Error: {e.Message}");
                }
            }
        }
    }

    void Update()
    {
        if (applyOnce && robotPoseApplied)
            return;

        while (robotMatrixQueue.TryDequeue(out var flat16))
        {
            // 🔹 Log the raw matrix values before applying
            Debug.Log($"[RobotBaseInitialPoseReceiver] 📥 Received robot_matrix (first 4) = " +
                      $"{flat16[0]:F4}, {flat16[1]:F4}, {flat16[2]:F4}, {flat16[3]:F4}");

            ApplyMatrixToRobotBase(flat16);
            robotPoseApplied = true;
            // If you want to keep updating when new matrices arrive, remove this line.
        }

        if (NetMQManager.IsShutdownRequested)
            ShutdownNetMQ();
    }

    private void ApplyMatrixToRobotBase(float[] flat16)
    {
        if (flat16 == null || flat16.Length != 16 || robotBase == null)
        {
            Debug.LogWarning("[RobotBaseInitialPoseReceiver] ⚠️ Invalid matrix or missing robotBase.");
            return;
        }

        Matrix4x4 mat = new Matrix4x4();
        for (int i = 0; i < 16; i++)
            mat[i] = flat16[i];

        // 🔹 Optional: full matrix log (comment out if too spammy)
        string matStr = string.Join(", ", flat16.Select(v => v.ToString("F4")));
        Debug.Log($"[RobotBaseInitialPoseReceiver] 📐 Raw robot_matrix = [{matStr}]");

        // Basic validity: last row should be (0,0,0,1)
        if (!IsValidMatrix(mat))
        {
            Debug.LogWarning("[RobotBaseInitialPoseReceiver] ⚠️ Received non-homogeneous matrix, ignoring rotation.");
        }

        Vector3 localPos = mat.GetColumn(3);
        Quaternion localRot = IsValidMatrix(mat) ? mat.rotation : Quaternion.identity;

        Debug.Log($"[RobotBaseInitialPoseReceiver] 🔎 Parsed localPos={localPos}, rotEuler={localRot.eulerAngles}");

        // Python sends position in the calibrated world frame (= WorldRoot local space).
        // Convert to Unity world space using the assigned WorldRoot transform.
        if (worldRoot != null)
            Debug.Log($"[RobotCalibPose] WorldRoot pos={worldRoot.position} rot={worldRoot.eulerAngles}");
        else
            Debug.LogWarning("[RobotCalibPose] worldRoot is NULL — using raw localPos as world pos");

        Vector3    worldPos = worldRoot != null ? worldRoot.TransformPoint(localPos) : localPos;
        Quaternion worldRot = worldRoot != null ? worldRoot.rotation * localRot      : localRot;

        Debug.Log($"[RobotCalibPose] → worldPos={worldPos} worldRot={worldRot.eulerAngles}");

        if (robotBase.TryGetComponent<ArticulationBody>(out var ab))
        {
            ab.TeleportRoot(worldPos, worldRot);
        }
        else
        {
            robotBase.SetPositionAndRotation(worldPos, worldRot);
        }

        Debug.Log($"[RobotCalibPose] ✅ Applied. robotBase now at pos={robotBase.transform.position} rot={robotBase.transform.eulerAngles} active={robotBase.gameObject.activeInHierarchy}");
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
        if (!isRunning || hasShutdown) return;
        hasShutdown = true;

        Debug.Log("[RobotBaseInitialPoseReceiver] 🔻 Shutting down...");

        try
        {
            isRunning = false;
            if (receiveThread != null && receiveThread.IsAlive)
                receiveThread.Join(1000);

            subscriber?.Dispose();
            subscriber = null;

            NetMQManager.UnregisterReceiver();
            Debug.Log("[RobotBaseInitialPoseReceiver] ✅ Shutdown complete");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[RobotBaseInitialPoseReceiver] ⚠️ Shutdown exception: " + e.Message);
        }
    }

    private void OnDestroy() => ShutdownNetMQ();
    private void OnApplicationQuit() => ShutdownNetMQ();

    [Serializable]
    public class RobotMatrixData
    {
        public float[] robot_matrix;
    }
}
