using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using Newtonsoft.Json;
using System.Collections.Generic;

public class RobotJointNetMQReceiver : MonoBehaviour
{
    public ArticulationBody[] robotJoints;
    [SerializeField] private int port = 5001;
    [SerializeField] private bool incomingAnglesAreRadians = true;

    private Thread receiveThread;
    private SubscriberSocket subscriber;
    private bool isRunning = false;
    private bool hasShutdown = false;
    private float[] latestAngles;

    private float _logTimer = 0f;
    private const float LogInterval = 1f;

    void Start()
    {
        latestAngles = new float[robotJoints.Length];
        isRunning = true;

        receiveThread = new Thread(ListenerLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        NetMQManager.RegisterReceiver();
        Debug.Log("📡 RobotJointNetMQReceiver started.");
    }

    void Update()
    {
        // Global shutdown request
        if (NetMQManager.IsShutdownRequested)
        {
            ShutdownNetMQ();
            return;
        }

        if (!isRunning) return;

        lock (latestAngles)
        {
            for (int i = 0; i < robotJoints.Length && i < latestAngles.Length; i++)
            {
                float angleRad = incomingAnglesAreRadians
                    ? latestAngles[i]
                    : latestAngles[i] * Mathf.Deg2Rad;
                robotJoints[i].jointPosition = new ArticulationReducedSpace(angleRad);
            }

            _logTimer += Time.deltaTime;
            if (_logTimer >= LogInterval)
            {
                _logTimer = 0f;
                var sb = new System.Text.StringBuilder("[RobotJoints] angles (rad | deg): ");
                for (int i = 0; i < latestAngles.Length; i++)
                {
                    float deg = incomingAnglesAreRadians ? latestAngles[i] * Mathf.Rad2Deg : latestAngles[i];
                    sb.Append($"j{i}={latestAngles[i]:F3}r/{deg:F1}° ");
                }
                Debug.Log(sb.ToString());

                if (robotJoints.Length > 0 && robotJoints[0] != null)
                {
                    var root = robotJoints[0];
                    Debug.Log($"[RobotJoints] root pos={root.transform.position} rot={root.transform.eulerAngles} active={root.gameObject.activeInHierarchy}");
                }
            }
        }
    }

    private void ListenerLoop()
    {
        AsyncIO.ForceDotNet.Force();

        try
        {
            subscriber = new SubscriberSocket();
            string address = $"tcp://0.0.0.0:{port}";
            subscriber.Bind(address);
            subscriber.Subscribe("");

            Debug.Log($"🛰️ Bound for joint data on {address}");

            while (isRunning)
            {
                if (subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string msg))
                {
                    Debug.Log($"[RobotJoints] 📥 Raw msg received: {msg}");
                    JointValueMessage parsed = JsonConvert.DeserializeObject<JointValueMessage>(msg);
                    if (parsed?.joint_values != null)
                    {
                        lock (latestAngles)
                        {
                            for (int i = 0; i < parsed.joint_values.Count && i < latestAngles.Length; i++)
                                latestAngles[i] = parsed.joint_values[i];
                        }
                        Debug.Log($"[RobotJoints] ✅ Applied {parsed.joint_values.Count} joint values");
                    }
                    else
                    {
                        Debug.LogWarning("[RobotJoints] ⚠️ Parsed message has null joint_values");
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("⚠️ Joint ZMQ receive error: " + e.Message);
        }
        finally
        {
            try
            {
                subscriber?.Close();
                subscriber?.Dispose();
                subscriber = null;
                // Debug.Log("✅ NetMQ subscriber socket closed.");
            }
            catch (Exception e)
            {
                Debug.LogWarning("⚠️ Error while disposing NetMQ socket: " + e.Message);
            }
        }
    }

    public void ShutdownNetMQ()
    {
        if (!isRunning || hasShutdown) return;
        hasShutdown = true;

        Debug.Log("🔻 Shutting down RobotJointNetMQReceiver...");

        try
        {
            isRunning = false;
            if (receiveThread != null && receiveThread.IsAlive)
                receiveThread.Join(1000); // timeout safeguard

            NetMQManager.UnregisterReceiver();
            Debug.Log("✅ RobotJointNetMQReceiver shutdown complete");
        }
        catch (Exception e)
        {
            Debug.LogWarning("⚠️ Shutdown exception: " + e.Message);
        }
    }

    private void OnDestroy() => ShutdownNetMQ();
    private void OnApplicationQuit() => ShutdownNetMQ();
}
