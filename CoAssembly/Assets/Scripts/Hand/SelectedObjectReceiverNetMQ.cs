using UnityEngine;
using UnityEngine.Events;

using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;

using System;
using System.Collections.Concurrent;
using System.Threading;
// If you get an error on AsyncIO.ForceDotNet.Force(), uncomment this:
// using AsyncIO;

public class SelectedObjectReceiverNetMQ : MonoBehaviour
{
    // ============================================================
    // Inspector
    // ============================================================

    [Header("NetMQ")]
    [SerializeField] private int port = 5006; // MUST match python bind port

    [Header("Debug")]
    public bool enableLogs = true;
    public bool logRaw = false;

    // ============================================================
    // UnityEvents (optional, for wiring to other components)
    // ============================================================

    [Serializable]
    public class SelectedObjectUnityEvent : UnityEvent<string, float> { }

    [Header("Events (optional)")]
    [Tooltip("Invoked on main thread when a selected_object msg is received: (keyLower, timestampSecondsFloat)")]
    public SelectedObjectUnityEvent onSelectedObject;

    // ============================================================
    // Latest received (read-only)
    // ============================================================

    [Header("Latest received (read-only)")]
    [SerializeField] private string latestKeyLower = "";
    [SerializeField] private float latestTimestamp = 0f;
    [SerializeField] private int receivedCount = 0;

    public string LatestKeyLower => latestKeyLower;
    public float LatestTimestamp => latestTimestamp;
    public int ReceivedCount => receivedCount;

    // ============================================================
    // NetMQ thread/queue
    // ============================================================

    private SubscriberSocket subscriber;
    private Thread receiveThread;
    private volatile bool isRunning = false;
    private bool hasShutdown = false;

    private readonly ConcurrentQueue<SelectedObjectMsg> messageQueue = new ConcurrentQueue<SelectedObjectMsg>();

    [Serializable]
    public class SelectedObjectMsg
    {
        public string type;        // "selected_object"
        public double timestamp;   // seconds
        public string key;         // "cube1"
        public float progress01;   // optional (python may include)
    }

    // ============================================================
    // Unity lifecycle
    // ============================================================

    private void Start()
    {
        StartNetMQ();
    }

    private void Update()
    {
        // Keep only latest message per frame (recommended)
        SelectedObjectMsg last = null;
        while (messageQueue.TryDequeue(out var m)) last = m;
        if (last == null) return;

        string t = (last.type ?? "").Trim().ToLower();
        if (!string.IsNullOrEmpty(t) && t != "selected_object")
            return;

        string key = (last.key ?? "").Trim().ToLower();
        float ts = (float)last.timestamp;

        latestKeyLower = key;
        latestTimestamp = ts;
        receivedCount++;

        if (enableLogs)
            Debug.Log($"[SelectedObjectReceiver] RX: key={key}, ts={ts:0.000}", this);

        // Output ONLY via event (no visuals here)
        onSelectedObject?.Invoke(key, ts);
    }

    private void OnDestroy() => ShutdownNetMQ();
    private void OnApplicationQuit() => ShutdownNetMQ();

    // ============================================================
    // NetMQ lifecycle
    // ============================================================

    private void StartNetMQ()
    {
        if (isRunning) return;

        hasShutdown = false;
        isRunning = true;

        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();

        ZEDNetMQManager.RegisterReceiver();
        Debug.Log($"[SelectedObjectReceiver] 📡 NetMQ enabled tcp://localhost:{port}", this);
    }

    private void StopNetMQ()
    {
        if (!isRunning) return;

        try
        {
            isRunning = false;

            if (receiveThread != null && receiveThread.IsAlive)
                receiveThread.Join(1000);

            subscriber?.Dispose();
            subscriber = null;

            ZEDNetMQManager.UnregisterReceiver();
        }
        catch (Exception e)
        {
            Debug.LogWarning("[SelectedObjectReceiver] ⚠️ StopNetMQ exception: " + e.Message, this);
        }

        Debug.Log($"[SelectedObjectReceiver] 📴 NetMQ disabled tcp://localhost:{port}", this);
    }

    public void ShutdownNetMQ()
    {
        if (hasShutdown) return;
        hasShutdown = true;
        StopNetMQ();
    }

    // ============================================================
    // NetMQ receive loop
    // ============================================================

    private void ReceiveLoop()
    {
        AsyncIO.ForceDotNet.Force();

        using (subscriber = new SubscriberSocket())
        {
            subscriber.Connect($"tcp://localhost:{port}");
            subscriber.Options.ReceiveHighWatermark = 1;
            subscriber.Subscribe("");

            while (isRunning)
            {
                if (subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string raw))
                {
                    if (string.IsNullOrEmpty(raw)) continue;

                    try
                    {
                        var msg = JsonConvert.DeserializeObject<SelectedObjectMsg>(raw);
                        if (msg != null) messageQueue.Enqueue(msg);

                        if (logRaw)
                            Debug.Log($"[SelectedObjectReceiver] RAW: {raw}", this);
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning($"[SelectedObjectReceiver] ⚠️ JSON parse error: {e.Message}", this);
                    }
                }
            }
        }
    }
}
