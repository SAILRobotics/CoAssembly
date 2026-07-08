using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using UnityEngine;

/// <summary>
/// Receives the secondary relock cubes' world poses from Python (port 5017) and
/// positions each authored "Secondary Reset Anchor" on its physical ArUco marker,
/// using the prescan registration as the single source of truth.
///
/// Python (main_with_robot._RelockCubePublisher) publishes:
///   {"cubes":[{"id":104,"position":[x,y,z],"rotation_xyzw":[x,y,z,w]}, ...]}
/// where position/rotation are already in Unity coordinates, expressed in the
/// WorldRoot (reference-marker) frame.
///
/// Each cube is matched by id to the interactable whose ToolClickPublisher.toolId
/// equals that id, and set as a LOCAL pose (the cubes are children of WorldRoot, so
/// they drift-correct with it). Ids without a matching cube are ignored — so you can
/// prescan more markers than you have authored cubes, or vice versa.
///
/// Attach this to a single GameObject in the scene (e.g. WorldRoot or a manager).
/// </summary>
public class RelockCubePoseReceiver : MonoBehaviour
{
    [SerializeField] private int port = 5017;

    [Serializable] private class Cube { public int id; public float[] position; public float[] rotation_xyzw; }
    [Serializable] private class Payload { public List<Cube> cubes; }

    private SubscriberSocket _sock;
    private Thread _thread;
    private volatile bool _running;
    private readonly ConcurrentQueue<Payload> _queue = new();
    private readonly Dictionary<int, Transform> _byId = new();

    private void Start()
    {
        // Map marker id -> cube transform via each cube's ToolClickPublisher.
        foreach (var pub in FindObjectsOfType<ToolClickPublisher>(true))
            _byId[pub.toolId] = pub.transform;
        Debug.Log($"[RelockCubePose] 🟢 Start; {_byId.Count} interactable id(s) known.");

        AsyncIO.ForceDotNet.Force();
        _sock = new SubscriberSocket();
        _sock.Bind($"tcp://0.0.0.0:{port}");
        _sock.Subscribe("");
        NetMQManager.RegisterReceiver();

        _running = true;
        _thread = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
        Debug.Log($"[RelockCubePose] 📡 SUB bound on tcp://0.0.0.0:{port}");
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
                    if (p?.cubes != null) _queue.Enqueue(p);
                }
            }
            catch (TerminatingException) { break; }
            catch (ObjectDisposedException) { break; }
            catch (Exception e) { if (_running) Debug.LogWarning($"[RelockCubePose] {e.Message}"); }
        }
    }

    private void Update()
    {
        while (_queue.TryDequeue(out var p))
        {
            foreach (var c in p.cubes)
            {
                if (c.position == null || c.position.Length < 3) continue;
                if (!_byId.TryGetValue(c.id, out var t) || t == null) continue;
                t.localPosition = new Vector3(c.position[0], c.position[1], c.position[2]);
                if (c.rotation_xyzw != null && c.rotation_xyzw.Length >= 4)
                    t.localRotation = new Quaternion(
                        c.rotation_xyzw[0], c.rotation_xyzw[1], c.rotation_xyzw[2], c.rotation_xyzw[3]);
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
