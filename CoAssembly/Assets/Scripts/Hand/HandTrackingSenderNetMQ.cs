using UnityEngine;
using NetMQ;
using NetMQ.Sockets;

public class HandTrackingSenderNetMQ : MonoBehaviour
{
    [Header("NetMQ")]
    [SerializeField] private int port = 5570;
    [SerializeField] private string bindHost = "*"; // "*" = all interfaces, "localhost" = local only

    [Header("Debug")]
    [SerializeField] private bool log = true;

    private PublisherSocket pubSocket;
    private int sentCount = 0;
    private float logTimer = 0f;

    void Start()
    {
        AsyncIO.ForceDotNet.Force();

        if (port <= 0 || port > 65535)
        {
            Debug.LogError($"[HandTrackingSenderNetMQ] Invalid port: {port}. Must be 1..65535", this);
            enabled = false;
            return;
        }

        pubSocket = new PublisherSocket();
        string endpoint = $"tcp://{bindHost}:{port}";
        pubSocket.Bind(endpoint);

        NetMQManager.RegisterSender();
        if (log) Debug.Log($"[HandTrackingSenderNetMQ] 📡 Sender bound on {endpoint}", this);
    }

    public void SendJson(string json)
    {
        if (pubSocket == null) return;
        if (string.IsNullOrEmpty(json)) return;

        try
        {
            pubSocket.SendFrame(json);
            sentCount++;
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("[HandTrackingSenderNetMQ] Send error: " + e.Message, this);
        }
    }

    private void Update()
    {
        if (!log) return;

        logTimer += Time.deltaTime;
        if (logTimer < 1f) return;

        Debug.Log($"[HandTrackingSenderNetMQ] Sent {sentCount} hand frames total on port {port}", this);
        logTimer = 0f;
    }

    private void OnDestroy() => ShutdownNetMQ();
    private void OnApplicationQuit() => ShutdownNetMQ();

    private void ShutdownNetMQ()
    {
        if (log) Debug.Log("[HandTrackingSenderNetMQ] 🔻 Shutting down sender...", this);

        try
        {
            pubSocket?.Dispose();
            pubSocket = null;
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("[HandTrackingSenderNetMQ] ⚠️ Dispose error: " + e.Message, this);
        }

        NetMQManager.UnregisterSender();
        if (log) Debug.Log("[HandTrackingSenderNetMQ] ✅ Sender shutdown complete", this);
    }
}
