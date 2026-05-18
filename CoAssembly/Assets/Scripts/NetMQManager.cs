using UnityEngine;
using NetMQ;

public static class NetMQManager
{
    public static bool IsShutdownRequested = false;

    private static int activeReceivers = 0;
    private static int activeSenders = 0;
    private static bool cleanedUp = false;

    public static void RegisterReceiver()
    {
        activeReceivers++;
        cleanedUp = false;
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
        cleanedUp = false;
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
        if (cleanedUp) return;

        if (activeReceivers == 0 && activeSenders == 0)
        {
            cleanedUp = true;
            Debug.Log("🧹 Final cleanup — calling NetMQConfig.Cleanup(false)");
            NetMQConfig.Cleanup(false);
        }
    }
}