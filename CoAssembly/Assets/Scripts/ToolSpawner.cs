using UnityEngine;
using UnityEngine.Rendering;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using Oculus.Interaction;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Receives tool layout from Python (port 5011) and spawns/despawns tool
/// prefabs at the specified world positions.  When the JSON on disk changes
/// Python republishes and this script respawns automatically.
///
/// Setup
/// -----
/// 1. Create one prefab per tool type with its 4 scripts, collider, renderer.
/// 2. Add entries to the toolPrefabs array in the Inspector: type name → prefab.
/// 3. Assign leftRayInteractor and rightRayInteractor from the player rig.
/// </summary>
public class ToolSpawner : MonoBehaviour
{
    [Serializable]
    public class ToolPrefabEntry
    {
        public int        id;
        public string     type;   // informational only
        public GameObject prefab;
    }

    [Header("Prefabs (id → prefab)")]
    public ToolPrefabEntry[] toolPrefabs = new ToolPrefabEntry[0];

    [Header("Parent transform (drag WorldRoot here)")]
    public Transform worldRoot;

    [Header("Rig references — injected into HandAwareInteractable on spawn")]
    public RayInteractor leftRayInteractor;
    public RayInteractor rightRayInteractor;

    [Header("NetMQ")]
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int    port = 5011;

    [Header("Pegboard visualization")]
    [Tooltip("Draw spawned pegboard tools/parts as subtle transparent boxes with crisp outlines.")]
    [SerializeField] private bool useBlendedBoxVisualization = true;

    [Tooltip("Face alpha for the blended cube volume. 0 = no fill at all (outline-only boxes); " +
             "this is also a hard ceiling — ToolColorReceiver clamps incoming face alpha to this " +
             "value, so leaving it at 0 keeps faces invisible no matter what color state (hover/" +
             "selected/highlight) Python sends.")]
    [Range(0f, 1f)] [SerializeField] private float faceAlpha = 0f;

    [Tooltip("Edge alpha for the cube outline.")]
    [Range(0f, 1f)] [SerializeField] private float edgeAlpha = 0.9f;

    [Tooltip("Default visualization color until ToolColorReceiver receives the Python color on port 5010.")]
    [SerializeField] private Color wireframeDefaultColor = Color.white;

    // ── Internal ─────────────────────────────────────────────────────────────
    [Serializable] private class ToolData
    {
        public int     id;
        public string  type;
        public string  category;   // "tool" or "part" (informational, colour applied by Python)
        public float[] position;
        public float[] rotation_xyzw;
        public float[] size;
    }
    [Serializable] private class Payload { public List<ToolData> tools; }

    private Thread          receiveThread;
    private volatile bool   isRunning   = false;
    private bool            hasShutdown = false;
    private SubscriberSocket subscriber;
    private readonly ConcurrentQueue<List<ToolData>> dataQueue = new();

    private readonly Dictionary<int, GameObject> spawnedTools = new();

    private const float EdgeWidth = 0.003f;

    private struct SpawnedToolVisual
    {
        public Renderer[] edgeRenderers;
        public Renderer faceRenderer;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void Start()
    {
        NetMQManager.RegisterReceiver();
        isRunning     = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();
        Debug.Log($"[ToolSpawner] Listening on tcp://0.0.0.0:{port}");
    }

    void Update()
    {
        List<ToolData> latest = null;
        while (dataQueue.TryDequeue(out var batch)) latest = batch;
        if (latest != null) ApplyLayout(latest);

        if (NetMQManager.IsShutdownRequested) Shutdown();
    }

    // ── Receive ───────────────────────────────────────────────────────────────
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
                            if (p?.tools != null) dataQueue.Enqueue(p.tools);
                        }
                    }
                    catch (TerminatingException)    { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e) { if (isRunning) Debug.LogWarning("[ToolSpawner] " + e.Message); }
                }
            }
        }
        catch (Exception e) { if (isRunning) Debug.LogWarning("[ToolSpawner] Outer: " + e.Message); }
    }

    // ── Spawn / despawn ───────────────────────────────────────────────────────
    private void ApplyLayout(List<ToolData> tools)
    {
        // Collect ids present in the new layout
        var newIds = new HashSet<int>();
        foreach (var t in tools) newIds.Add(t.id);

        // Despawn tools no longer in the layout
        var toRemove = new List<int>();
        foreach (var kv in spawnedTools)
            if (!newIds.Contains(kv.Key)) toRemove.Add(kv.Key);
        foreach (var id in toRemove)
        {
            Destroy(spawnedTools[id]);
            spawnedTools.Remove(id);
        }

        // Spawn or update each tool
        foreach (var t in tools)
        {
            if (!spawnedTools.TryGetValue(t.id, out var go))
            {
                var prefab = FindPrefab(t.id);
                if (prefab == null)
                {
                    Debug.LogWarning($"[ToolSpawner] No prefab for id {t.id} (type '{t.type}')");
                    continue;
                }
                go = Instantiate(prefab, worldRoot);
                SpawnedToolVisual visual = useBlendedBoxVisualization ? SetupBlendedBoxVisual(go) : default;
                InjectReferences(go, t.id, visual);
                spawnedTools[t.id] = go;
            }

            // Apply transform — localPosition/Rotation so positions are relative to WorldRoot
            if (t.position != null && t.position.Length == 3)
                go.transform.localPosition = new Vector3(t.position[0], t.position[1], t.position[2]);

            if (t.rotation_xyzw != null && t.rotation_xyzw.Length == 4)
                go.transform.localRotation = new Quaternion(t.rotation_xyzw[0], t.rotation_xyzw[1],
                                                            t.rotation_xyzw[2], t.rotation_xyzw[3]);

            if (t.size != null && t.size.Length == 3)
                go.transform.localScale = new Vector3(t.size[0], t.size[1], t.size[2]);
        }
    }

    private GameObject FindPrefab(int id)
    {
        foreach (var entry in toolPrefabs)
            if (entry.id == id) return entry.prefab;
        return null;
    }

    private void InjectReferences(GameObject go, int toolId, SpawnedToolVisual visual)
    {
        // Inject rig refs into HandAwareInteractable
        var hai = go.GetComponent<HandAwareInteractable>();
        if (hai != null)
        {
            hai._leftInteractor  = leftRayInteractor;
            hai._rightInteractor = rightRayInteractor;
        }

        // Set tool IDs
        var tcp = go.GetComponent<ToolClickPublisher>();
        if (tcp != null) tcp.toolId = toolId;

        var tcr = go.GetComponent<ToolColorReceiver>();
        if (tcr != null)
        {
            if (visual.edgeRenderers != null && visual.edgeRenderers.Length > 0)
                tcr.ConfigureVisual(toolId, visual.edgeRenderers, visual.faceRenderer, faceAlpha);
            else if (visual.faceRenderer != null)
                tcr.ConfigureVisual(toolId, visual.faceRenderer, null, faceAlpha);
            else
                tcr.toolId = toolId;
        }
    }

    private SpawnedToolVisual SetupBlendedBoxVisual(GameObject go)
    {
        var sourceRenderers = go.GetComponentsInChildren<Renderer>(true);
        foreach (var renderer in sourceRenderers)
        {
            if (renderer != null)
                renderer.enabled = false;
        }

        Transform old = go.transform.Find("ToolVisualization");
        if (old != null)
            Destroy(old.gameObject);

        var root = new GameObject("ToolVisualization");
        root.transform.SetParent(go.transform, false);

        Color faceColor = wireframeDefaultColor;
        faceColor.a = faceAlpha;
        var face = new GameObject("Faces");
        face.transform.SetParent(root.transform, false);
        var faceFilter = face.AddComponent<MeshFilter>();
        var faceRenderer = face.AddComponent<MeshRenderer>();
        faceRenderer.shadowCastingMode = ShadowCastingMode.Off;
        faceRenderer.receiveShadows = false;
        faceRenderer.material = CreateTransparentFaceMaterial("ToolFaceMaterial", faceColor);
        faceFilter.mesh = BuildUnitCubeFaceMesh();

        Renderer[] edgeRenderers = null;
        if (edgeAlpha > 0f)
        {
            Color edgeColor = wireframeDefaultColor;
            edgeColor.a = edgeAlpha;
            var edges = new GameObject("Edges");
            edges.transform.SetParent(root.transform, false);
            edgeRenderers = AddCubeEdgeLines(edges.transform, edgeColor);
        }

        return new SpawnedToolVisual
        {
            edgeRenderers = edgeRenderers,
            faceRenderer = faceRenderer,
        };
    }

    private Renderer[] AddCubeEdgeLines(Transform parent, Color color)
    {
        var vertices = UnitCubeVertices();
        var pairs = new int[]
        {
            0,1, 1,2, 2,3, 3,0,
            4,5, 5,6, 6,7, 7,4,
            0,4, 1,5, 2,6, 3,7,
        };
        var mat = CreateEdgeMaterial("ToolEdgeMaterial", color);
        var renderers = new Renderer[pairs.Length / 2];
        for (int i = 0; i < pairs.Length; i += 2)
        {
            var go = new GameObject("Edge");
            go.transform.SetParent(parent, false);
            var line = go.AddComponent<LineRenderer>();
            line.useWorldSpace = false;
            line.positionCount = 2;
            line.SetPosition(0, vertices[pairs[i]]);
            line.SetPosition(1, vertices[pairs[i + 1]]);
            line.widthMultiplier = EdgeWidth;
            line.numCapVertices = 0;
            line.numCornerVertices = 0;
            line.shadowCastingMode = ShadowCastingMode.Off;
            line.receiveShadows = false;
            line.material = mat;
            renderers[i / 2] = line;
        }
        return renderers;
    }

    private static Vector3[] UnitCubeVertices()
    {
        return new Vector3[]
        {
            new(-0.5f, -0.5f, -0.5f),
            new( 0.5f, -0.5f, -0.5f),
            new( 0.5f,  0.5f, -0.5f),
            new(-0.5f,  0.5f, -0.5f),
            new(-0.5f, -0.5f,  0.5f),
            new( 0.5f, -0.5f,  0.5f),
            new( 0.5f,  0.5f,  0.5f),
            new(-0.5f,  0.5f,  0.5f),
        };
    }
    private static Material CreateTransparentFaceMaterial(string name, Color color)
    {
        var shader = Shader.Find("Universal Render Pipeline/Lit")
                  ?? Shader.Find("Universal Render Pipeline/Unlit")
                  ?? Shader.Find("Standard");
        var mat = new Material(shader) { name = name };
        mat.renderQueue = (int)RenderQueue.Transparent;
        mat.SetOverrideTag("RenderType", "Transparent");

        if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
        if (mat.HasProperty("_Blend")) mat.SetFloat("_Blend", 0f);
        if (mat.HasProperty("_AlphaClip")) mat.SetFloat("_AlphaClip", 0f);
        if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        if (mat.HasProperty("_SrcBlendAlpha")) mat.SetFloat("_SrcBlendAlpha", (float)BlendMode.One);
        if (mat.HasProperty("_DstBlendAlpha")) mat.SetFloat("_DstBlendAlpha", (float)BlendMode.OneMinusSrcAlpha);
        if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);
        if (mat.HasProperty("_Cull")) mat.SetFloat("_Cull", (float)CullMode.Off);
        if (mat.HasProperty("_ReceiveShadows")) mat.SetFloat("_ReceiveShadows", 0f);

        mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.DisableKeyword("_ALPHAMODULATE_ON");
        SetMaterialColor(mat, color);
        return mat;
    }

    private static Material CreateEdgeMaterial(string name, Color color)
    {
        var shader = Shader.Find("Universal Render Pipeline/Unlit")
                  ?? Shader.Find("Unlit/Color")
                  ?? Shader.Find("Standard");
        var mat = new Material(shader) { name = name };
        mat.renderQueue = (int)RenderQueue.Transparent;
        mat.SetOverrideTag("RenderType", "Transparent");
        if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
        if (mat.HasProperty("_SrcBlend")) mat.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
        if (mat.HasProperty("_DstBlend")) mat.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
        if (mat.HasProperty("_ZWrite")) mat.SetFloat("_ZWrite", 0f);
        mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        SetMaterialColor(mat, color);
        return mat;
    }

    private static void SetMaterialColor(Material mat, Color color)
    {
        if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    private static Mesh BuildUnitCubeFaceMesh()
    {
        var vertices = new Vector3[]
        {
            new(-0.5f, -0.5f, -0.5f),
            new( 0.5f, -0.5f, -0.5f),
            new( 0.5f,  0.5f, -0.5f),
            new(-0.5f,  0.5f, -0.5f),
            new(-0.5f, -0.5f,  0.5f),
            new( 0.5f, -0.5f,  0.5f),
            new( 0.5f,  0.5f,  0.5f),
            new(-0.5f,  0.5f,  0.5f),
        };
        var triangles = new int[]
        {
            0,2,1, 0,3,2,
            4,5,6, 4,6,7,
            0,1,5, 0,5,4,
            2,3,7, 2,7,6,
            1,2,6, 1,6,5,
            0,4,7, 0,7,3,
        };
        var mesh = new Mesh { name = "ToolBlendedCubeFaces" };
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }
    private void Shutdown()
    {
        if (hasShutdown) return;
        hasShutdown = true;
        isRunning   = false;
        subscriber?.Close();
        subscriber?.Dispose();
        subscriber = null;
        if (receiveThread?.IsAlive == true) receiveThread.Join(1000);
        NetMQManager.UnregisterReceiver();
        Debug.Log("[ToolSpawner] Shutdown complete");
    }

    private void OnDestroy()         => Shutdown();
    private void OnApplicationQuit() => Shutdown();
}
