using UnityEngine;
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

    [Header("Rig references — injected into HandAwareInteractable on spawn")]
    public RayInteractor leftRayInteractor;
    public RayInteractor rightRayInteractor;

    [Header("NetMQ")]
    [SerializeField] private string host = "127.0.0.1";
    [SerializeField] private int    port = 5011;

    // ── Internal ─────────────────────────────────────────────────────────────
    [Serializable] private class ToolData
    {
        public int     id;
        public string  type;
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
                go = Instantiate(prefab);
                InjectReferences(go, t.id);
                spawnedTools[t.id] = go;
            }

            // Apply transform
            if (t.position != null && t.position.Length == 3)
                go.transform.position = new Vector3(t.position[0], t.position[1], t.position[2]);

            if (t.rotation_xyzw != null && t.rotation_xyzw.Length == 4)
                go.transform.rotation = new Quaternion(t.rotation_xyzw[0], t.rotation_xyzw[1],
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

    private void InjectReferences(GameObject go, int toolId)
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
        if (tcr != null) tcr.toolId = toolId;
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
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
