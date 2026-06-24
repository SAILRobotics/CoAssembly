using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System.Threading;
using System;
using System.Globalization;

public class RobotiqGripperNetMQController_NoLimits : MonoBehaviour
{
    [System.Serializable]
    public class MimicJoint
    {
        public ArticulationBody joint;
        public float multiplier = 1f;
    }

    [Header("Joints")]
    public ArticulationBody controlJoint;
    public MimicJoint[] mimicJoints;

    [Header("Input")]
    [Tooltip("If true, the received float is in radians.")]
    public bool inputIsRadians = true;

    [Header("NetMQ")]
    public int GripperPort = 5004;

    [Header("Debug")]
    public bool logSetAngle = false;

    // ── Ghost mode (ABs disabled — Transform-based joint animation) ───────────
    // When GhostGripperReceiver disables all ArticulationBodies, this script
    // detects it in Start() and switches to driving joints via localRotation.
    // The AB's anchorRotation is read before the component is disabled so we
    // know the correct rotation axis for each joint.
    private bool         ghostMode = false;
    private Vector3[]    jointAxes;   // effective revolute axis in parent local space
    private Quaternion[] restRots;    // localRotation at rest (angle = 0)

    private Thread receiveThread;
    private bool isRunning = true;
    private bool hasShutdown = false;
    private SubscriberSocket subscriber;
    private float? pendingGripperAngle = null;
    private readonly object stateLock = new object();

    void Start()
    {
        if (controlJoint == null)
        {
            Debug.LogError("❌ controlJoint not assigned.");
            enabled = false;
            return;
        }

        // If the AB is disabled we are on the ghost gripper prefab.
        // Cache the joint axes and rest rotations from the (disabled) AB config,
        // then use Transform.localRotation to animate fingers instead of physics.
        if (!controlJoint.enabled)
        {
            ghostMode = true;
            int n = 1 + (mimicJoints != null ? mimicJoints.Length : 0);
            jointAxes = new Vector3[n];
            restRots  = new Quaternion[n];

            CacheJoint(controlJoint, 0);
            for (int i = 0; i < (mimicJoints != null ? mimicJoints.Length : 0); i++)
                if (mimicJoints[i].joint != null)
                    CacheJoint(mimicJoints[i].joint, i + 1);
        }

        SetGripperAngle(0f);

        receiveThread = new Thread(ReceiveGripperAngle) { IsBackground = true };
        receiveThread.Start();
        NetMQManager.RegisterReceiver();
        Debug.Log($"📡 Gripper controller started on port {GripperPort}" +
                  (ghostMode ? " [ghost/Transform mode]" : ""));
    }

    // Cache the revolute axis and rest localRotation for one joint.
    // anchorRotation rotates the AB's joint-frame X-axis into the body's local frame,
    // so the effective revolute axis in parent space = anchorRotation * Vector3.right.
    private void CacheJoint(ArticulationBody ab, int idx)
    {
        jointAxes[idx] = ab.anchorRotation * Vector3.right;
        restRots[idx]  = ab.transform.localRotation;
    }

    void Update()
    {
        float? angle = null;
        lock (stateLock)
        {
            if (pendingGripperAngle.HasValue)
            {
                angle = pendingGripperAngle.Value;
                pendingGripperAngle = null;
            }
        }
        if (angle.HasValue)
            SetGripperAngle(angle.Value);

        if (NetMQManager.IsShutdownRequested)
            ShutdownNetMQ();
    }

    public void SetGripperAngle(float angleDeg)
    {
        if (controlJoint == null) return;

        if (!ghostMode)
        {
            // Normal mode — drive via ArticulationBody joint positions.
            float angleRad = angleDeg * Mathf.Deg2Rad;
            controlJoint.jointPosition = new ArticulationReducedSpace(angleRad);
            if (mimicJoints != null)
                foreach (var mj in mimicJoints)
                    if (mj.joint != null)
                        mj.joint.jointPosition = new ArticulationReducedSpace(mj.multiplier * angleRad);
        }
        else if (jointAxes != null)
        {
            // Ghost mode — drive via Transform.localRotation.
            ApplyAngle(controlJoint.transform, 0, angleDeg);
            for (int i = 0; i < (mimicJoints != null ? mimicJoints.Length : 0); i++)
                if (mimicJoints[i]?.joint != null)
                    ApplyAngle(mimicJoints[i].joint.transform, i + 1,
                               mimicJoints[i].multiplier * angleDeg);
        }

        if (logSetAngle)
            Debug.Log($"[Gripper] {angleDeg:F2}°{(ghostMode ? " (ghost)" : "")}");
    }

    private void ApplyAngle(Transform t, int idx, float angleDeg)
    {
        // Rotate by angleDeg around the cached joint axis, relative to the rest pose.
        t.localRotation = Quaternion.AngleAxis(angleDeg, jointAxes[idx]) * restRots[idx];
    }

    private void ReceiveGripperAngle()
    {
        AsyncIO.ForceDotNet.Force();
        using (subscriber = new SubscriberSocket())
        {
            string address = $"tcp://0.0.0.0:{GripperPort}";
            subscriber.Bind(address);
            subscriber.Subscribe("");

            var inv = CultureInfo.InvariantCulture;
            Debug.Log($"🔗 Gripper listening at {address}");

            while (isRunning)
            {
                try
                {
                    if (subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string msg))
                    {
                        if (float.TryParse(msg, NumberStyles.Float, inv, out float rawValue))
                        {
                            float angleDeg = inputIsRadians ? rawValue * Mathf.Rad2Deg : rawValue;
                            lock (stateLock) { pendingGripperAngle = angleDeg; }
                        }
                        else
                        {
                            Debug.LogWarning($"⚠️ Invalid gripper angle received: '{msg}'");
                        }
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning("⚠️ Gripper NetMQ receive error: " + e.Message);
                }
            }
        }
    }

    private void ShutdownNetMQ()
    {
        if (!isRunning || hasShutdown) return;
        hasShutdown = true;

        isRunning = false;
        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Join(1000);

        try { subscriber?.Dispose(); subscriber = null; }
        catch (Exception e) { Debug.LogWarning("⚠️ Error during Gripper socket disposal: " + e.Message); }

        NetMQManager.UnregisterReceiver();
        Debug.Log("✅ Gripper NetMQ shutdown complete");
    }

    private void OnDestroy() => ShutdownNetMQ();
    private void OnApplicationQuit() => ShutdownNetMQ();
}
