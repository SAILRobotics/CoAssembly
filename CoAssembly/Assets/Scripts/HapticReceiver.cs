using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Receives haptic commands from Python (port 5007) and plays them on
/// Meta Quest 3 controllers via OVRInput.SetControllerVibration().
///
/// Attach to any GameObject in your scene.
///
/// Python usage:
///   haptic.vibrate("left",  amplitude=1.0, frequency=0.5, duration_ms=150)
///   haptic.vibrate("right", amplitude=0.5, frequency=1.0, duration_ms=80)
///   haptic.vibrate("both",  amplitude=0.8, frequency=0.0, duration_ms=300)
///
/// OVR parameters:
///   frequency : 0.0 = low rumble,  1.0 = high buzz
///   amplitude : 0.0 = off,         1.0 = maximum
/// </summary>
public class HapticReceiver : MonoBehaviour
{
    [Header("NetMQ")]
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int    port = 5007;

    // ── Internal ─────────────────────────────────────────────────────
    private Thread receiveThread;
    private volatile bool isRunning   = false;
    private bool          hasShutdown = false;
    private SubscriberSocket subscriber;

    private readonly ConcurrentQueue<HapticCommand> queue = new();

    [Serializable]
    private class HapticCommand
    {
        public string controller;   // "left" | "right" | "both"
        public float  amplitude;    // 0.0 – 1.0
        public float  frequency;    // 0.0 – 1.0
        public int    duration_ms;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────
    void Start()
    {
        NetMQManager.RegisterReceiver();
        isRunning = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();
        Debug.Log($"[HapticReceiver] Listening on tcp://0.0.0.0:{port}");
    }

    private void ReceiveLoop()
    {
        AsyncIO.ForceDotNet.Force();
        try
        {
            using (subscriber = new SubscriberSocket())
            {
                subscriber.Bind($"tcp://0.0.0.0:{port}");
                subscriber.Subscribe("");

                while (isRunning)
                {
                    try
                    {
                        if (subscriber.TryReceiveFrameString(
                                TimeSpan.FromMilliseconds(100), out string msg))
                        {
                            var cmd = JsonConvert.DeserializeObject<HapticCommand>(msg);
                            if (cmd != null) queue.Enqueue(cmd);
                        }
                    }
                    catch (TerminatingException)    { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e) { if (isRunning) Debug.LogWarning("[HapticReceiver] " + e.Message); }
                }
            }
        }
        catch (Exception e) { if (isRunning) Debug.LogWarning("[HapticReceiver] Outer: " + e.Message); }
    }

    void Update()
    {
        while (queue.TryDequeue(out var cmd))
            StartCoroutine(PlayHaptic(cmd));

        if (NetMQManager.IsShutdownRequested)
            Shutdown();
    }

    // ── Haptic playback ───────────────────────────────────────────────
    private IEnumerator PlayHaptic(HapticCommand cmd)
    {
        OVRInput.Controller ctrl = cmd.controller switch
        {
            "left"  => OVRInput.Controller.LTouch,
            "right" => OVRInput.Controller.RTouch,
            _       => OVRInput.Controller.Touch,   // Touch = both LTouch + RTouch
        };

        OVRInput.SetControllerVibration(cmd.frequency, cmd.amplitude, ctrl);
        yield return new WaitForSeconds(cmd.duration_ms / 1000f);
        OVRInput.SetControllerVibration(0f, 0f, ctrl);
    }

    // ── Shutdown ──────────────────────────────────────────────────────
    private void Shutdown()
    {
        if (hasShutdown) return;
        hasShutdown = true;
        isRunning = false;

        // Stop any ongoing vibration on shutdown
        OVRInput.SetControllerVibration(0f, 0f, OVRInput.Controller.Touch);

        subscriber?.Close();
        subscriber?.Dispose();
        subscriber = null;

        if (receiveThread?.IsAlive == true)
            receiveThread.Join(1000);

        NetMQManager.UnregisterReceiver();
        Debug.Log("[HapticReceiver] Shutdown complete");
    }

    private void OnDestroy()         => Shutdown();
    private void OnApplicationQuit() => Shutdown();
}
