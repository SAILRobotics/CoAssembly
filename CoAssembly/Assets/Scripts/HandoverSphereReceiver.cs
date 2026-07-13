using System;
using System.Collections.Concurrent;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using UnityEngine;

/// <summary>
/// Receives the handover target sphere's world position from Python (port 5018)
/// and places THIS GameObject (a sphere parented under WorldRoot) at it, expressed
/// in the WorldRoot / marker-100 frame — so it drift-corrects with the rest of the
/// scene, exactly like the relock cubes.
///
/// Python (main_with_robot._HandoverSpherePublisher) publishes:
///   {"position":[x,y,z], "visible":true|false}
/// with position already in Unity coordinates. `visible` toggles the renderer; the
/// sphere is shown when a tool is grasped and hidden once the robot reaches the
/// delivery point.
///
/// Attach to a sphere GameObject that is a CHILD of WorldRoot. Give it a small
/// localScale (e.g. 0.08) and an unlit / emissive material so it reads clearly in AR.
/// </summary>
public class HandoverSphereReceiver : MonoBehaviour
{
    [SerializeField] private int port = 5018;
    [SerializeField] private Renderer targetRenderer;   // defaults to this GameObject's Renderer

    [Serializable] private class Payload { public float[] position; public bool visible; }

    private SubscriberSocket _sock;
    private Thread _thread;
    private volatile bool _running;
    private readonly ConcurrentQueue<Payload> _queue = new();

    private void Start()
    {
        if (targetRenderer == null) targetRenderer = GetComponent<Renderer>();
        if (targetRenderer != null) targetRenderer.enabled = false;   // hidden until first "visible" payload

        AsyncIO.ForceDotNet.Force();
        _sock = new SubscriberSocket();
        _sock.Bind($"tcp://0.0.0.0:{port}");
        _sock.Subscribe("");
        NetMQManager.RegisterReceiver();

        _running = true;
        _thread = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
        Debug.Log($"[HandoverSphere] 📡 SUB bound on tcp://0.0.0.0:{port}");
    }

    private void ReceiveLoop()
    {
        while (_running)
        {
            try
            {
                if (_sock.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string msg))
                {
                    var p = JsonConvert.DeserializeObject<Payload>(msg);
                    if (p != null) _queue.Enqueue(p);
                }
            }
            catch (TerminatingException) { break; }
            catch (ObjectDisposedException) { break; }
            catch (Exception e) { if (_running) Debug.LogWarning($"[HandoverSphere] {e.Message}"); }
        }
    }

    private void Update()
    {
        while (_queue.TryDequeue(out var p))
        {
            if (p.visible && p.position != null && p.position.Length >= 3)
            {
                transform.localPosition = new Vector3(p.position[0], p.position[1], p.position[2]);
                if (targetRenderer != null) targetRenderer.enabled = true;
            }
            else if (targetRenderer != null)
            {
                targetRenderer.enabled = false;
            }
        }
    }

    private void OnDestroy()
    {
        _running = false;
        if (_thread != null && _thread.IsAlive) _thread.Join(500);
        if (_sock != null)
        {
            try { _sock.Close(); _sock.Dispose(); } catch { }
            _sock = null;
            NetMQManager.UnregisterReceiver();
        }
    }
}
