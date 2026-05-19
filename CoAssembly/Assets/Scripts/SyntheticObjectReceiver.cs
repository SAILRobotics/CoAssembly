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

    [Serializable] private class ObjectData
    {
        public int     id;
        public float[] position;
        public float[] rotation_xyzw;
        public float[] size;
    }
    [Serializable] private class Payload { public List<ObjectData> objects; }

    // ── Lifecycle ─────────────────────────────────────────────────────
    void Start()
    {
        _visualReady  = new bool[objects.Length];
        _lastSeenTime = new float[objects.Length];
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

            if (obj.position != null && obj.position.Length == 3)
                tf.localPosition = new Vector3(obj.position[0],
                                               obj.position[1],
                                               obj.position[2]);

            if (obj.rotation_xyzw != null && obj.rotation_xyzw.Length == 4)
                tf.localRotation = new Quaternion(obj.rotation_xyzw[0],
                                                  obj.rotation_xyzw[1],
                                                  obj.rotation_xyzw[2],
                                                  obj.rotation_xyzw[3]);

            if (applyScale && obj.size != null && obj.size.Length == 3)
                tf.localScale = new Vector3(obj.size[0], obj.size[1], obj.size[2]);
        }
    }

    // ── Visual (called once per object on first activation) ──────────
    private void SetupVisual(int id, Transform tf)
    {
        Color c = id < ObjectColors.Length ? ObjectColors[id] : Color.white;

        // Face — instance the user's template material, tint it
        if (templateMaterial != null)
        {
            var rend = tf.GetComponent<Renderer>();
            if (rend != null)
            {
                var mat = new Material(templateMaterial);
                float a = templateMaterial.color.a;   // preserve the alpha you set
                // Cover both Built-in (_Color) and URP (_BaseColor)
                mat.SetColor("_Color",     new Color(c.r, c.g, c.b, a));
                mat.SetColor("_BaseColor", new Color(c.r, c.g, c.b, a));
                rend.material = mat;
            }
        }

        // Edges — thin wireframe via MeshTopology.Lines
        var old = tf.Find("Edges");
        if (old != null) Destroy(old.gameObject);
        AddEdges(tf, new Color(c.r, c.g, c.b, 1f));
    }

    private void AddEdges(Transform parent, Color color)
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
