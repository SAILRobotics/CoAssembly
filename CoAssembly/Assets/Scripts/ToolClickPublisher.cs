using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using Newtonsoft.Json;

public class ToolClickPublisher : MonoBehaviour
{
    [Header("Tool Identity")]
    [SerializeField] private int toolId;

    [Header("NetMQ")]
    [SerializeField] private int port = 5009;

    private static PublisherSocket sharedSocket;
    private static int refCount = 0;
    private static readonly object lockObj = new object();

    [Serializable]
    private class ClickMessage
    {
        public int tool_id;
        public string event_type;
    }

    private void Start()
    {
        Debug.Log($"[ToolClickPublisher] 🟢 Start() called on GameObject '{gameObject.name}', toolId={toolId}");
        lock (lockObj)
        {
            if (sharedSocket == null)
            {
                AsyncIO.ForceDotNet.Force();
                sharedSocket = new PublisherSocket();
                sharedSocket.Bind($"tcp://0.0.0.0:{port}");
                NetMQManager.RegisterSender();
                Debug.Log($"[ToolClickPublisher] 📡 Bound shared PUB on tcp://0.0.0.0:{port}");
            }
            refCount++;
        }
    }

    public void OnSelected()
    {
        SendEvent("selected");
    }

    public void OnHoverEnter()
    {
        SendEvent("hover_enter");
    }

    public void OnHoverExit()
    {
        SendEvent("hover_exit");
    }

    private void SendEvent(string eventType)
    {
        var msg = new ClickMessage { tool_id = toolId, event_type = eventType };
        try
        {
            sharedSocket?.SendFrame(JsonConvert.SerializeObject(msg));
            Debug.Log($"[ToolClickPublisher:{toolId}] 📤 Sent {eventType}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[ToolClickPublisher] Send failed: {e.Message}");
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