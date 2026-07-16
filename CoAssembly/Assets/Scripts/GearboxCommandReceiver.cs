using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Newtonsoft.Json;

/// <summary>
/// Drives the gearbox visualization from an external Python process over NetMQ.
/// Attach this to the "Gearbox Assembly Named" root GameObject.
///
/// Commands (JSON, sent by gearbox_control.py):
///   {"command":"row","row":N}      → show only parts whose name contains "RowN"
///   {"command":"toggle","part":".."}→ flip a single part between highlight and original color
///   {"command":"show_all"}          → make every row visible again
///   {"command":"reset"}             → clear all color highlights
///
/// Mirrors the established receiver pattern in ToolColorReceiver.cs: bind a SubscriberSocket,
/// receive JSON on a background thread into a ConcurrentQueue, apply on the main thread in Update().
/// </summary>
public class GearboxCommandReceiver : MonoBehaviour
{
    [Header("NetMQ")]
    [SerializeField] private int port = 5019;

    [Header("Visual")]
    [Tooltip("Color a part turns to when toggled on. Default = yellow.")]
    [SerializeField] private Color highlightColor = new Color(1f, 0.92f, 0.016f, 1f);

    [Tooltip("Root of the gearbox hierarchy. Leave empty to use this GameObject's transform.")]
    [SerializeField] private Transform gearboxRoot;

    [Serializable]
    private class GearboxCommand
    {
        public string command;   // "row" | "toggle" | "show_all" | "reset"
        public int    row;       // used when command == "row"
        public string part;      // used when command == "toggle"
    }

    // ── Per-part cached state ────────────────────────────────────────────────
    private class PartEntry
    {
        public GameObject           go;
        public Renderer             renderer;
        public MaterialPropertyBlock block;
        public int                  colorID;
        public Color                originalColor;
        public bool                 highlighted;
    }

    private readonly List<PartEntry>                 parts        = new();
    private readonly Dictionary<string, PartEntry>   partsByName  = new();
    private readonly Dictionary<string, PartEntry>   partsByLower = new();

    // ── Socket + dispatcher ──────────────────────────────────────────────────
    private SubscriberSocket socket;
    private Thread receiveThread;
    private volatile bool running = false;
    private readonly ConcurrentQueue<GearboxCommand> pending = new();

    private void Start()
    {
        if (gearboxRoot == null) gearboxRoot = transform;

        BuildPartIndex();

        AsyncIO.ForceDotNet.Force();
        socket = new SubscriberSocket();
        socket.Bind($"tcp://0.0.0.0:{port}");
        socket.Subscribe("");
        NetMQManager.RegisterReceiver();

        running = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();

        Debug.Log($"[GearboxCommandReceiver] 📡 SUB bound on tcp://0.0.0.0:{port}, " +
                  $"indexed {parts.Count} parts under '{gearboxRoot.name}'.");
    }

    private void BuildPartIndex()
    {
        // A "part" = a transform whose name contains "Row" AND has a Renderer.
        // This naturally excludes the root itself and any non-part nodes, so we
        // never SetActive(false) the GameObject this script lives on.
        foreach (Transform t in gearboxRoot.GetComponentsInChildren<Transform>(true))
        {
            if (!t.name.Contains("Row")) continue;

            Renderer r = t.GetComponent<Renderer>();
            if (r == null) continue;

            Material mat = r.sharedMaterial;
            if (mat == null) continue;

            // colorID may be 0 if the shader exposes no known base-color property;
            // the part still participates in row show/hide, only its color toggle is disabled.
            int colorID = ResolveColorID(mat);
            if (colorID == 0)
                Debug.LogWarning($"[GearboxCommandReceiver] '{t.name}' has no known base-color " +
                                 $"property (shader '{mat.shader.name}'); color toggle disabled for it.");

            var entry = new PartEntry
            {
                go            = t.gameObject,
                renderer      = r,
                block         = new MaterialPropertyBlock(),
                colorID       = colorID,
                originalColor = colorID != 0 ? mat.GetColor(colorID) : Color.white,
                highlighted   = false,
            };

            parts.Add(entry);
            partsByName[t.name]              = entry;
            partsByLower[t.name.ToLower()]   = entry;
        }
    }

    private void ReceiveLoop()
    {
        try
        {
            while (running)
            {
                try
                {
                    if (socket.TryReceiveFrameString(
                            TimeSpan.FromMilliseconds(100), out string message))
                    {
                        var cmd = JsonConvert.DeserializeObject<GearboxCommand>(message);
                        if (cmd != null && !string.IsNullOrEmpty(cmd.command))
                        {
                            Debug.Log($"[GearboxCommandReceiver] 📥 RX {message}");
                            pending.Enqueue(cmd);
                        }
                    }
                }
                catch (TerminatingException) { break; }
                catch (ObjectDisposedException) { break; }
                catch (Exception e)
                {
                    if (running)
                        Debug.LogWarning($"[GearboxCommandReceiver] {e.Message}");
                }
            }
        }
        catch (Exception e)
        {
            if (running)
                Debug.LogWarning($"[GearboxCommandReceiver] Outer: {e.Message}");
        }
    }

    private void Update()
    {
        while (pending.TryDequeue(out GearboxCommand cmd))
        {
            switch (cmd.command)
            {
                case "row":       ShowOnlyRow(cmd.row); break;
                case "show_all":  ShowAll();            break;
                case "toggle":    TogglePart(cmd.part); break;
                case "reset":     ResetHighlights();    break;
                default:
                    Debug.LogWarning($"[GearboxCommandReceiver] Unknown command '{cmd.command}'");
                    break;
            }
        }
    }

    private void ShowOnlyRow(int row)
    {
        string tag = $"Row{row}";
        foreach (var p in parts)
            p.go.SetActive(p.go.name.Contains(tag));
        Debug.Log($"[GearboxCommandReceiver] 👁 Showing only {tag}");
    }

    private void ShowAll()
    {
        foreach (var p in parts)
            p.go.SetActive(true);
        Debug.Log("[GearboxCommandReceiver] 👁 Showing all rows");
    }

    private void TogglePart(string name)
    {
        if (string.IsNullOrEmpty(name)) return;

        if (!partsByName.TryGetValue(name, out PartEntry entry))
            partsByLower.TryGetValue(name.ToLower(), out entry);

        if (entry == null)
        {
            Debug.LogWarning($"[GearboxCommandReceiver] No part named '{name}'");
            return;
        }
        if (entry.colorID == 0)
        {
            Debug.LogWarning($"[GearboxCommandReceiver] '{name}' has no color property; cannot highlight.");
            return;
        }

        entry.highlighted = !entry.highlighted;
        ApplyColor(entry, entry.highlighted ? highlightColor : entry.originalColor);
        Debug.Log($"[GearboxCommandReceiver] 🎨 '{name}' → " +
                  (entry.highlighted ? "highlight" : "original"));
    }

    private void ResetHighlights()
    {
        int cleared = 0;
        foreach (var p in parts)
        {
            if (!p.highlighted) continue;
            p.highlighted = false;
            ApplyColor(p, p.originalColor);
            cleared++;
        }
        Debug.Log($"[GearboxCommandReceiver] ♻ Reset {cleared} highlight(s)");
    }

    // Candidate base-color property names, in priority order:
    //   baseColorFactor → glTFast shader graph (com.unity.cloud.gltfast) — the gearbox parts
    //   _BaseColor      → URP Lit
    //   _Color          → Built-in / legacy
    private static readonly string[] ColorProps = { "baseColorFactor", "_BaseColor", "_Color" };

    private static int ResolveColorID(Material mat)
    {
        foreach (string prop in ColorProps)
            if (mat.HasProperty(prop))
                return Shader.PropertyToID(prop);
        return 0;   // none found
    }

    private static void ApplyColor(PartEntry entry, Color c)
    {
        entry.renderer.GetPropertyBlock(entry.block);
        entry.block.SetColor(entry.colorID, c);
        entry.renderer.SetPropertyBlock(entry.block);
    }

    private void OnDestroy()
    {
        running = false;

        if (socket != null)
        {
            try { socket.Close(); socket.Dispose(); } catch { }
            socket = null;
        }

        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Join(500);
        receiveThread = null;

        NetMQManager.UnregisterReceiver();
    }
}
