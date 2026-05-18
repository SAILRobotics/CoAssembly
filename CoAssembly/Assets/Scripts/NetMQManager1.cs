using UnityEngine;
using NetMQ;

public static class ZEDNetMQManager
{
    public static bool IsShutdownRequested = false;

    private static int activeReceivers = 0;
    private static int activeSenders = 0;

    public static void RegisterReceiver()
    {
        activeReceivers++;
        Debug.Log($"📥 NetMQ receiver registered. Active receivers: {activeReceivers}");
    }

    public static void UnregisterReceiver()
    {
        activeReceivers = Mathf.Max(0, activeReceivers - 1);
        Debug.Log($"📤 NetMQ receiver unregistered. Remaining: {activeReceivers}");
        CheckAndCleanup();
    }

    public static void RegisterSender()
    {
        activeSenders++;
        Debug.Log($"📡 NetMQ sender registered. Active senders: {activeSenders}");
    }

    public static void UnregisterSender()
    {
        activeSenders = Mathf.Max(0, activeSenders - 1);
        Debug.Log($"📡 NetMQ sender unregistered. Remaining: {activeSenders}");
        CheckAndCleanup();
    }

    private static void CheckAndCleanup()
    {
        if (activeReceivers <= 0 && activeSenders <= 0)
        {
            Debug.Log("🧹 Final cleanup — calling NetMQConfig.Cleanup()");
            NetMQConfig.Cleanup();
        }
    }
}
