using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Barebones receiver for the workholding AR-box test harness (workholding_testing.py).
///
/// Binds WORKHOLDING_BOX_PORT (5026) — its OWN port, distinct from GripStateReceiver's 5012 so the
/// two never clash — and consumes the SAME message schema as GripStateReceiver
/// (grip_state, box_pos, box_rot_xyzw, box_size), but — unlike GripStateReceiver, which only
/// reveals the box on an ISDK grab — this ALWAYS shows the box at the received pose. That lets
/// the harness park the box at each of its 10 test poses with no grabbing involved.
///
/// Inspector setup:
///   arBox     — the AR box GameObject (a 1x1x1 m unit cube so box_size is literal metres),
///               placed as a child of worldRoot
///   worldRoot — WorldRoot transform (same object WorldRoot.cs / ToolSpawner drive)
///
/// Mirrors the socket / background-thread / ConcurrentQueue / Update lifecycle and the
/// NetMQManager register/unregister pattern from GripStateReceiver.cs.
/// </summary>
public class WorkholdingBoxReceiver : MonoBehaviour
{
    [Header("AR box")]
    public GameObject arBox;

    [Header("Parent")]
    public Transform worldRoot;

    [Header("NetMQ")]
    [SerializeField] private int port = 5026;   // WORKHOLDING_BOX_PORT (not 5012 / GRIP_STATE_PORT)

    [Serializable]
    private class BoxMessage
    {
        public string  grip_state;      // ignored — the box is always shown
        public float[] box_pos;
        public float[] box_rot_xyzw;
        public float[] box_size;
    }

    private class PoseData
    {
        public Vector3    pos;
        public Quaternion rot;
        public Vector3    size;
    }

    private SubscriberSocket _socket;
    private Thread           _thread;
    private volatile bool    _running;
    private readonly ConcurrentQueue<PoseData> _queue = new();
    private Renderer[]       _renderers;
    private bool             _loggedShow;

    private void Start()
    {
        // Hide until the first pose arrives by toggling the RENDERER(s), not the GameObject:
        // this component may live on arBox itself, and SetActive(false) on our own GameObject
        // would stop Update() from ever running (so the box could never be shown again).
        if (arBox != null)
        {
            _renderers = arBox.GetComponentsInChildren<Renderer>(true);
            SetVisible(false);
        }

        NetMQManager.RegisterReceiver();
        _running = true;
        _thread  = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
    }

    private void ReceiveLoop()
    {
        AsyncIO.ForceDotNet.Force();
        try
        {
            using (_socket = new SubscriberSocket())
            {
                _socket.Bind($"tcp://0.0.0.0:{port}");
                _socket.Subscribe("");
                Debug.Log($"[WorkholdingBoxReceiver] SUB bound on tcp://0.0.0.0:{port}");
                while (_running)
                {
                    try
                    {
                        if (_socket.TryReceiveFrameString(
                                TimeSpan.FromMilliseconds(100), out string msg))
                        {
                            var d = JsonConvert.DeserializeObject<BoxMessage>(msg);
                            if (d == null || d.box_pos == null || d.box_rot_xyzw == null
                                || d.box_size == null) continue;
                            _queue.Enqueue(new PoseData
                            {
                                pos  = new Vector3(d.box_pos[0], d.box_pos[1], d.box_pos[2]),
                                rot  = new Quaternion(d.box_rot_xyzw[0], d.box_rot_xyzw[1],
                                                      d.box_rot_xyzw[2], d.box_rot_xyzw[3]),
                                size = new Vector3(d.box_size[0], d.box_size[1], d.box_size[2]),
                            });
                        }
                    }
                    catch (TerminatingException)    { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e) { if (_running) Debug.LogWarning("[WorkholdingBoxReceiver] " + e.Message); }
                }
            }
        }
        catch (Exception e) { if (_running) Debug.LogWarning("[WorkholdingBoxReceiver] Outer: " + e.Message); }
    }

    private void Update()
    {
        PoseData latest = null;
        while (_queue.TryDequeue(out var d)) latest = d;
        if (latest == null || arBox == null) return;

        // The pose arrives in world coordinates; WorldRoot IS the world origin in Unity, so apply
        // it as a local pose under worldRoot (same convention as GripStateReceiver / ToolSpawner).
        if (worldRoot != null && arBox.transform.parent != worldRoot)
            arBox.transform.SetParent(worldRoot, false);
        arBox.transform.localPosition = latest.pos;
        arBox.transform.localRotation = latest.rot;
        arBox.transform.localScale    = latest.size;
        SetVisible(true);

        if (!_loggedShow)
        {
            _loggedShow = true;
            Debug.Log($"[WorkholdingBoxReceiver] First box pose applied — localPos="
                      + $"{arBox.transform.localPosition}  scale={latest.size}  "
                      + $"renderers={( _renderers != null ? _renderers.Length : 0)}");
        }
    }

    private void SetVisible(bool visible)
    {
        if (_renderers == null) return;
        foreach (var r in _renderers)
            if (r != null) r.enabled = visible;
    }

    private void OnDestroy()
    {
        _running = false;
        _socket?.Close();
        if (_thread?.IsAlive == true) _thread.Join(500);
        NetMQManager.UnregisterReceiver();
    }
}
