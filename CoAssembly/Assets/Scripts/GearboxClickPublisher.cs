using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using Newtonsoft.Json;

/// <summary>
/// Publishes gearbox part / UI clicks to Python over NetMQ, keyed by the GameObject's NAME
/// (parts follow the {Type}Row{N}{Side} convention; the checkbox / reset objects are named
/// "__checkbox__" / "__reset__"). This is the gearbox-specific twin of ToolClickPublisher —
/// same shared-socket lifecycle, but a dedicated port so it never crosses the robot arm's
/// tool-click pipeline, and it carries the part NAME instead of a numeric tool_id.
///
/// Bound (server) side; Python CONNECTs a zmq.SUB to tcp://&lt;this-machine&gt;:5023.
/// </summary>
public class GearboxClickPublisher : MonoBehaviour
{
    [Header("NetMQ")]
    // 5023 (was 5020, which collided with ROBOT_CMD_PORT). See main_setting.GEARBOX_CLICK_PORT.
    [SerializeField] private int port = 5023;

    private static PublisherSocket sharedSocket;
    private static int refCount = 0;
    private static readonly object lockObj = new object();

    [Serializable]
    private class ClickMessage
    {
        public string name;
        public string event_type;
    }

    private void Start()
    {
        lock (lockObj)
        {
            if (sharedSocket == null)
            {
                AsyncIO.ForceDotNet.Force();
                sharedSocket = new PublisherSocket();
                sharedSocket.Bind($"tcp://0.0.0.0:{port}");
                NetMQManager.RegisterSender();
                Debug.Log($"[GearboxClickPublisher] 📡 Bound shared PUB on tcp://0.0.0.0:{port}");
            }
            refCount++;
        }
    }

    /// <summary>Send a "selected" click carrying this GameObject's name.</summary>
    public void SendSelected() => SendEvent("selected");

    public void SendEvent(string eventType)
    {
        var msg = new ClickMessage { name = gameObject.name, event_type = eventType };
        try
        {
            sharedSocket?.SendFrame(JsonConvert.SerializeObject(msg));
            Debug.Log($"[GearboxClickPublisher] 📤 Sent {eventType} '{gameObject.name}'");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[GearboxClickPublisher] Send failed: {e.Message}");
        }
    }

    private void OnDestroy()
    {
        lock (lockObj)
        {
            refCount = Mathf.Max(0, refCount - 1);
            if (refCount == 0 && sharedSocket != null)
            {
                sharedSocket.Close();
                sharedSocket.Dispose();
                sharedSocket = null;
                NetMQManager.UnregisterSender();
            }
        }
    }
}
