using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Receives synthetic-object poses from Python (port 5006).
///
/// Assign your own transparent material (alpha 0.4) to "Template Material"
/// in the Inspector — the script instances it per object and tints it with
/// the matching color. Edges are drawn with MeshTopology.Lines.
///
/// id 0 → objects[0] … id 9 → objects[9]
/// Objects not in the latest batch are deactivated.
/// </summary>
public class SyntheticObjectReceiver : MonoBehaviour
{
    [Header("Objects (drag Object0–Object9 from WorldRoot here)")]
    public Transform[] objects = new Transform[10];

    [Header("NetMQ")]
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int    port  = 5006;

    [Header("Visual")]
    [Tooltip("Assign a transparent material you created (alpha ~0.4). " +
             "The script will instance it and set the correct color per object.")]
    [SerializeField] private Material templateMaterial;
    [SerializeField] private bool applyScale = true;

    // Colors matching Python ids 0-9
    private static readonly Color[] ObjectColors =
    {
        new Color(1.00f, 0.10f, 0.10f), // 0 red
        new Color(0.10f, 0.85f, 0.10f), // 1 green
        new Color(0.10f, 0.20f, 1.00f), // 2 blue
        new Color(0.00f, 0.90f, 0.90f), // 3 cyan
        new Color(1.00f, 0.95f, 0.00f), // 4 yellow
        new Color(1.00f, 0.50f, 0.00f), // 5 orange
        new Color(0.40f, 0.75f, 1.00f), // 6 sky blue
        new Color(1.00f, 0.60f, 0.50f), // 7 melon
        new Color(0.60f, 0.10f, 0.90f), // 8 purple
        new Color(1.00f, 0.30f, 0.70f), // 9 pink
    };

    [Header("Per-object overrides")]
    [Tooltip("IDs listed here skip the auto wireframe-cube visual setup. " +
             "Use for objects that have their own visuals already in the scene (e.g. TCP sphere).")]
    public int[] skipAutoVisualIds = new int[0];

    [Header("Stability")]
    [Tooltip("Deactivate an object only after it has been absent for this many seconds. " +
             "Prevents single-frame message gaps from causing visible flicker.")]
    [SerializeField] private float deactivateTimeout = 0.5f;

    // ── Internal ─────────────────────────────────────────────────────
    private Thread receiveThread;
    private volatile bool isRunning   = false;
    private bool          hasShutdown = false;
    private SubscriberSocket subscriber;

    private readonly ConcurrentQueue<List<ObjectData>> dataQueue = new();
    private bool[]  _visualReady;
    private float[] _lastSeenTime;
    private List<Material>[] _faceMaterials;
    private Material[] _edgeMaterials;

    [Serializable] private class ObjectData
    {
        public int     id;
        public float[] position;
        public float[] rotation_xyzw;
        public float[] size;
        public float[] color;
    }
    [Serializable] private class Payload { public List<ObjectData> objects; }

    // ── Lifecycle ─────────────────────────────────────────────────────
    void Start()
    {
        _visualReady  = new bool[objects.Length];
        _lastSeenTime = new float[objects.Length];
        _faceMaterials = new List<Material>[objects.Length];
        _edgeMaterials = new Material[objects.Length];
        NetMQManager.RegisterReceiver();
        isRunning = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();
        Debug.Log($"[SyntheticObjectReceiver] Listening on tcp://0.0.0.0:{port}");
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
                            var p = JsonConvert.DeserializeObject<Payload>(msg);
                            if (p?.objects != null) dataQueue.Enqueue(p.objects);
                        }
                    }
                    catch (TerminatingException)    { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e) { if (isRunning) Debug.LogWarning("[SOR] " + e.Message); }
                }
            }
        }
        catch (Exception e) { if (isRunning) Debug.LogWarning("[SOR] Outer: " + e.Message); }
    }

    void Update()
    {
        List<ObjectData> latest = null;
        while (dataQueue.TryDequeue(out var b)) latest = b;
        if (latest != null) ApplyObjects(latest);

        // Deactivate objects that haven't been seen for deactivateTimeout seconds
        for (int i = 0; i < objects.Length; i++)
        {
            if (objects[i] != null && objects[i].gameObject.activeSelf &&
                Time.time - _lastSeenTime[i] > deactivateTimeout)
                objects[i].gameObject.SetActive(false);
        }

        if (NetMQManager.IsShutdownRequested) Shutdown();
    }

    // ── Apply ─────────────────────────────────────────────────────────
    private void ApplyObjects(List<ObjectData> batch)
    {
        foreach (var obj in batch)
        {
            if (obj.id < 0 || obj.id >= objects.Length) continue;
            var tf = objects[obj.id];
            if (tf == null) continue;

            _lastSeenTime[obj.id] = Time.time;

            if (!tf.gameObject.activeSelf)
                tf.gameObject.SetActive(true);

            if (!_visualReady[obj.id])
            {
                SetupVisual(obj.id, tf);
                _visualReady[obj.id] = true;
            }

            // Scene-authored visuals listed in skipAutoVisualIds (notably TCP
            // ID 3) keep their own color-control AND scale — they have their
            // own mesh authored at a fixed size, so port 5006's box-size
            // dimensions (meant for wireframe debug cubes) must not stomp it.
            // ToolColorReceiver on port 5010 owns TCP color.
            if (!ShouldSkipAutoVisual(obj.id))
                ApplyColor(obj);

            if (obj.position != null && obj.position.Length == 3)
                tf.localPosition = new Vector3(obj.position[0],
                                               obj.position[1],
                                               obj.position[2]);

            if (obj.rotation_xyzw != null && obj.rotation_xyzw.Length == 4)
                tf.localRotation = new Quaternion(obj.rotation_xyzw[0],
                                                  obj.rotation_xyzw[1],
                                                  obj.rotation_xyzw[2],
                                                  obj.rotation_xyzw[3]);

            if (applyScale && !ShouldSkipAutoVisual(obj.id)
                    && obj.size != null && obj.size.Length == 3)
                tf.localScale = new Vector3(obj.size[0], obj.size[1], obj.size[2]);
        }
    }

    // ── Visual (called once per object on first activation) ──────────
    private bool ShouldSkipAutoVisual(int id)
    {
        foreach (int skip in skipAutoVisualIds)
            if (id == skip) return true;
        return false;
    }

    private void SetupVisual(int id, Transform tf)
    {
        if (ShouldSkipAutoVisual(id))
            return;
        Color c = id < ObjectColors.Length ? ObjectColors[id] : Color.white;

        // Face — instance the user's template material, tint it
        if (templateMaterial != null)
        {
            var rend = tf.GetComponent<Renderer>();
            if (rend != null)
            {
                var mat = new Material(templateMaterial);
                rend.material = mat;
                _faceMaterials[id] = new List<Material> { mat };
            }
        }

        // Edges — thin wireframe via MeshTopology.Lines
        var old = tf.Find("Edges");
        if (old != null) Destroy(old.gameObject);
        _edgeMaterials[id] = AddEdges(tf, new Color(c.r, c.g, c.b, 1f));
    }

    private void ApplyColor(ObjectData obj)
    {
        Color fallback = obj.id < ObjectColors.Length
            ? ObjectColors[obj.id]
            : Color.white;
        float alpha = templateMaterial != null ? templateMaterial.color.a : 1f;
        var faces = _faceMaterials[obj.id];
        Color c = fallback;
        if (obj.color != null && obj.color.Length >= 3)
        {
            if (obj.color.Length >= 4) alpha = obj.color[3];
            c = new Color(obj.color[0], obj.color[1], obj.color[2], alpha);
        }
        else
        {
            c.a = alpha;
        }

        if (faces != null)
        {
            foreach (var face in faces)
            {
                face.SetColor("_Color", c);
                face.SetColor("_BaseColor", c);
            }
        }

        var edge = _edgeMaterials[obj.id];
        if (edge != null)
        {
            var edgeColor = new Color(c.r, c.g, c.b, 1f);
            edge.SetColor("_Color", edgeColor);
            edge.SetColor("_BaseColor", edgeColor);
        }
    }

    private Material AddEdges(Transform parent, Color color)
    {
        var go = new GameObject("Edges");
        go.transform.SetParent(parent, false);

        var mf = go.AddComponent<MeshFilter>();
        var mr = go.AddComponent<MeshRenderer>();
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows    = false;

        // Pick an unlit shader that works on this render pipeline
        var shader = Shader.Find("Unlit/Color")
                  ?? Shader.Find("Universal Render Pipeline/Unlit");
        var mat = new Material(shader);
        mat.SetColor("_Color",     color);
        mat.SetColor("_BaseColor", color);
        mr.material = mat;

        mf.mesh = BuildCubeEdgeMesh();
        return mat;
    }

    private static Mesh BuildCubeEdgeMesh()
    {
        var v = new Vector3[]
        {
            new(-0.5f, -0.5f, -0.5f), // 0
            new( 0.5f, -0.5f, -0.5f), // 1
            new( 0.5f,  0.5f, -0.5f), // 2
            new(-0.5f,  0.5f, -0.5f), // 3
            new(-0.5f, -0.5f,  0.5f), // 4
            new( 0.5f, -0.5f,  0.5f), // 5
            new( 0.5f,  0.5f,  0.5f), // 6
            new(-0.5f,  0.5f,  0.5f), // 7
        };
        var idx = new int[]
        {
            0,1, 1,2, 2,3, 3,0,   // back face
            4,5, 5,6, 6,7, 7,4,   // front face
            0,4, 1,5, 2,6, 3,7,   // connecting edges
        };
        var mesh = new Mesh();
        mesh.vertices = v;
        mesh.SetIndices(idx, MeshTopology.Lines, 0);
        return mesh;
    }

    // ── Shutdown ──────────────────────────────────────────────────────
    private void Shutdown()
    {
        if (hasShutdown) return;
        hasShutdown = true;
        isRunning = false;
        subscriber?.Close();
        subscriber?.Dispose();
        subscriber = null;
        if (receiveThread?.IsAlive == true) receiveThread.Join(1000);
        NetMQManager.UnregisterReceiver();
        Debug.Log("[SyntheticObjectReceiver] Shutdown complete");
    }

    private void OnDestroy()         => Shutdown();
    private void OnApplicationQuit() => Shutdown();
}
