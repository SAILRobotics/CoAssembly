using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Receives the robot workspace boundary from Python (port 5015) and draws it
/// as a 12-edge wireframe box, parented under WorldRoot. The wireframe is
/// built once from bounds_lo/bounds_hi (constant in world frame) and then
/// fades in/out smoothly based on dist_outside — 0 when the user's head and
/// both hands are inside the box (fully transparent), growing toward
/// maxAlpha as any of them moves further outside.
///
/// Inspector setup
/// ----------------
///   worldRoot — drag the WorldRoot transform here (same as every other
///               Python-driven object in the scene).
/// Everything else (geometry, material) is built at runtime — no prefab needed.
/// </summary>
public class WorkspaceBoundReceiver : MonoBehaviour
{
    [Header("Parent")]
    public Transform worldRoot;

    [Header("NetMQ")]
    [SerializeField] private int port = 5015;

    [Header("Appearance")]
    [SerializeField] private Color lineColor    = new Color(0.4f, 0.7f, 1.0f);
    [SerializeField] private float lineWidth    = 0.004f;
    [SerializeField] private float maxAlpha     = 0.5f;
    [Tooltip("dist_outside (metres) at which the wireframe reaches maxAlpha.")]
    [SerializeField] private float fadeDistance = 0.3f;
    [Tooltip("Higher = faster fade in/out.")]
    [SerializeField] private float fadeLerpSpeed = 4.0f;

    [Serializable]
    private class BoundsMessage
    {
        public float[] bounds_lo;
        public float[] bounds_hi;
        public float   dist_outside;
    }

    private SubscriberSocket _socket;
    private Thread           _thread;
    private volatile bool    _running;
    private readonly ConcurrentQueue<BoundsMessage> _queue = new();

    private LineRenderer[] _edges;
    private Material       _material;
    private Vector3        _lo, _hi;
    private bool           _boundsSet;
    private float          _currentAlpha;
    private float          _targetAlpha;

    // 12 edges of a box, expressed as (corner-bit-A, corner-bit-B) pairs.
    // Corner index bit0=x(hi), bit1=y(hi), bit2=z(hi) — 0 = lo on that axis.
    private static readonly (int, int)[] EdgePairs = new (int, int)[]
    {
        (0,1), (1,3), (3,2), (2,0),   // bottom face (z = lo)
        (4,5), (5,7), (7,6), (6,4),   // top face    (z = hi)
        (0,4), (1,5), (2,6), (3,7),   // verticals
    };

    private void Start()
    {
        if (worldRoot == null)
            Debug.LogWarning("[WorkspaceBoundReceiver] worldRoot not assigned — wireframe will use scene-root space.");

        transform.SetParent(worldRoot, false);
        transform.localPosition = Vector3.zero;
        transform.localRotation = Quaternion.identity;
        transform.localScale    = Vector3.one;

        BuildMaterial();
        BuildWireframe();

        NetMQManager.RegisterReceiver();
        _running = true;
        _thread  = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
        Debug.Log($"[WorkspaceBoundReceiver] Listening on tcp://0.0.0.0:{port}");
    }

    private void BuildMaterial()
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) shader = Shader.Find("Sprites/Default");
        _material = new Material(shader);
    }

    private void BuildWireframe()
    {
        _edges = new LineRenderer[EdgePairs.Length];
        for (int i = 0; i < EdgePairs.Length; i++)
        {
            var go = new GameObject($"edge_{i}");
            go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.material        = _material;
            lr.useWorldSpace   = false;
            lr.positionCount   = 2;
            lr.startWidth      = lineWidth;
            lr.endWidth        = lineWidth;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows  = false;
            SetEdgeAlpha(lr, 0f);
            _edges[i] = lr;
        }
    }

    private Vector3 Corner(int idx, Vector3 lo, Vector3 hi)
    {
        return new Vector3(
            (idx & 1) != 0 ? hi.x : lo.x,
            (idx & 2) != 0 ? hi.y : lo.y,
            (idx & 4) != 0 ? hi.z : lo.z);
    }

    private void ApplyBounds(Vector3 lo, Vector3 hi)
    {
        _lo = lo; _hi = hi; _boundsSet = true;
        for (int i = 0; i < EdgePairs.Length; i++)
        {
            var (a, b) = EdgePairs[i];
            _edges[i].SetPosition(0, Corner(a, lo, hi));
            _edges[i].SetPosition(1, Corner(b, lo, hi));
        }
    }

    private void SetEdgeAlpha(LineRenderer lr, float alpha)
    {
        Color c = lineColor; c.a = alpha;
        lr.startColor = c;
        lr.endColor   = c;
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
                while (_running)
                {
                    try
                    {
                        if (_socket.TryReceiveFrameString(
                                TimeSpan.FromMilliseconds(100), out string msg))
                        {
                            var d = JsonConvert.DeserializeObject<BoundsMessage>(msg);
                            if (d?.bounds_lo != null && d.bounds_lo.Length == 3
                                    && d.bounds_hi != null && d.bounds_hi.Length == 3)
                                _queue.Enqueue(d);
                        }
                    }
                    catch (TerminatingException)    { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e) { if (_running) Debug.LogWarning("[WorkspaceBoundReceiver] " + e.Message); }
                }
            }
        }
        catch (Exception e) { if (_running) Debug.LogWarning("[WorkspaceBoundReceiver] Outer: " + e.Message); }
    }

    private void Update()
    {
        BoundsMessage latest = null;
        while (_queue.TryDequeue(out var d)) latest = d;
        if (latest != null)
        {
            Vector3 lo = new Vector3(latest.bounds_lo[0], latest.bounds_lo[1], latest.bounds_lo[2]);
            Vector3 hi = new Vector3(latest.bounds_hi[0], latest.bounds_hi[1], latest.bounds_hi[2]);
            if (!_boundsSet || lo != _lo || hi != _hi)
                ApplyBounds(lo, hi);

            _targetAlpha = Mathf.Clamp01(latest.dist_outside / Mathf.Max(fadeDistance, 0.0001f)) * maxAlpha;
        }

        if (!Mathf.Approximately(_currentAlpha, _targetAlpha))
        {
            _currentAlpha = Mathf.Lerp(_currentAlpha, _targetAlpha, Time.deltaTime * fadeLerpSpeed);
            foreach (var lr in _edges) SetEdgeAlpha(lr, _currentAlpha);
        }

        if (NetMQManager.IsShutdownRequested) Shutdown();
    }

    private void Shutdown()
    {
        _running = false;
        _socket?.Close();
        if (_thread?.IsAlive == true) _thread.Join(500);
        NetMQManager.UnregisterReceiver();
    }

    private void OnDestroy()         => Shutdown();
    private void OnApplicationQuit() => Shutdown();
}
