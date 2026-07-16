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
/// Commands (JSON, sent by gearbox_control.py and gearbox_click_handler.py):
///   {"command":"row","row":N}                       → show only parts whose name contains "RowN"
///   {"command":"toggle","part":".."}                → flip a single part's highlight color
///   {"command":"show_all"}                          → make every row visible again
///   {"command":"reset"}                             → clear all color highlights
///   {"command":"show_subset","row":N,"types":[..]}  → show only the given part types of row N
///   {"command":"ui","show":bool,"row":N,"checked":bool}
///                                                   → show/hide + place the checkbox & reset X
///                                                     near row N; tint the checkbox
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

    [Header("State UI (checkbox + reset X)")]
    [Tooltip("The 'completed' checkbox object (a clickable 3D quad/cube). Optional.")]
    [SerializeField] private Transform checkboxObject;
    [Tooltip("The reset 'X' object (a clickable 3D quad/cube). Optional.")]
    [SerializeField] private Transform resetObject;
    [Tooltip("Renderer on the checkbox whose color reflects checked/unchecked. Optional.")]
    [SerializeField] private Renderer  checkboxRenderer;
    [SerializeField] private Color   checkedColor   = new Color(0.15f, 0.80f, 0.20f, 1f);  // green
    [SerializeField] private Color   uncheckedColor = new Color(0.80f, 0.80f, 0.80f, 1f);  // grey
    [Tooltip("Offset of the checkbox from the active row's bounds center; the X sits beside it.")]
    [SerializeField] private Vector3 uiOffset = new Vector3(0f, 0.15f, 0f);
    [SerializeField] private Vector3 resetOffsetFromCheckbox = new Vector3(0.08f, 0f, 0f);

    [Serializable]
    private class GearboxCommand
    {
        public string   command;   // "row"|"toggle"|"show_all"|"reset"|"show_subset"|"ui"
        public int      row;       // "row" | "show_subset" | "ui"
        public string   part;      // "toggle"
        public string[] types;     // "show_subset"
        public bool     show;      // "ui"
        [JsonProperty("checked")]
        public bool     isChecked; // "ui"
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
        public string               type;    // parsed prefix before "Row" (e.g. "GearRod", "Bearing")
        public int                  rowNum;  // parsed row number, or -1
    }

    private readonly List<PartEntry>                 parts        = new();
    private readonly Dictionary<string, PartEntry>   partsByName  = new();
    private readonly Dictionary<string, PartEntry>   partsByLower = new();

    // Checkbox tint state (resolved in Start).
    private int                   checkboxColorID;
    private MaterialPropertyBlock checkboxBlock;

    // ── Socket + dispatcher ──────────────────────────────────────────────────
    private SubscriberSocket socket;
    private Thread receiveThread;
    private volatile bool running = false;
    private readonly ConcurrentQueue<GearboxCommand> pending = new();

    private void Start()
    {
        if (gearboxRoot == null) gearboxRoot = transform;

        BuildPartIndex();

        if (checkboxRenderer != null && checkboxRenderer.sharedMaterial != null)
        {
            checkboxColorID = ResolveColorID(checkboxRenderer.sharedMaterial);
            checkboxBlock   = new MaterialPropertyBlock();
        }
        // Start with the UI hidden.
        if (checkboxObject) checkboxObject.gameObject.SetActive(false);
        if (resetObject)    resetObject.gameObject.SetActive(false);

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

            int rowIdx = t.name.IndexOf("Row");   // guaranteed >= 0 by the Contains check above

            var entry = new PartEntry
            {
                go            = t.gameObject,
                renderer      = r,
                block         = new MaterialPropertyBlock(),
                colorID       = colorID,
                originalColor = colorID != 0 ? mat.GetColor(colorID) : Color.white,
                highlighted   = false,
                type          = t.name.Substring(0, rowIdx),   // "GearRod", "Gear", "Bearing", ...
                rowNum        = ParseRow(t.name, rowIdx),
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
                case "row":         ShowOnlyRow(cmd.row);              break;
                case "show_all":    ShowAll();                         break;
                case "toggle":      TogglePart(cmd.part);              break;
                case "reset":       ResetHighlights();                 break;
                case "show_subset": ShowSubset(cmd.row, cmd.types);    break;
                case "ui":          ShowUi(cmd.row, cmd.show, cmd.isChecked); break;
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

    // Show only the given part types of a single row (a "state"); hide everything else.
    private void ShowSubset(int row, string[] types)
    {
        var set = new HashSet<string>(types ?? Array.Empty<string>());
        int shown = 0;
        foreach (var p in parts)
        {
            bool visible = p.rowNum == row && set.Contains(p.type);
            p.go.SetActive(visible);
            if (visible) shown++;
        }
        Debug.Log($"[GearboxCommandReceiver] 👁 Row{row} subset [{string.Join(",", set)}] → {shown} parts");
    }

    // Show/hide + place the checkbox and reset-X near the active row, and tint the checkbox.
    private void ShowUi(int row, bool show, bool isChecked)
    {
        if (!show)
        {
            if (checkboxObject) checkboxObject.gameObject.SetActive(false);
            if (resetObject)    resetObject.gameObject.SetActive(false);
            return;
        }

        // Combined bounds center of the row's currently-visible parts.
        Bounds? bounds = null;
        foreach (var p in parts)
        {
            if (p.rowNum != row || !p.go.activeInHierarchy) continue;
            if (bounds == null) bounds = p.renderer.bounds;
            else { Bounds b = bounds.Value; b.Encapsulate(p.renderer.bounds); bounds = b; }
        }
        Vector3 basePos = (bounds?.center ?? gearboxRoot.position) + uiOffset;

        if (checkboxObject)
        {
            checkboxObject.position = basePos;
            checkboxObject.gameObject.SetActive(true);
        }
        if (resetObject)
        {
            resetObject.position = basePos + resetOffsetFromCheckbox;
            resetObject.gameObject.SetActive(true);
        }
        ApplyCheckboxVisual(isChecked);
    }

    private void ApplyCheckboxVisual(bool isChecked)
    {
        if (checkboxRenderer == null || checkboxColorID == 0 || checkboxBlock == null) return;
        checkboxRenderer.GetPropertyBlock(checkboxBlock);
        checkboxBlock.SetColor(checkboxColorID, isChecked ? checkedColor : uncheckedColor);
        checkboxRenderer.SetPropertyBlock(checkboxBlock);
    }

    // Parse the row number that follows "Row" in a part name (e.g. "GearRodRow1" → 1).
    private static int ParseRow(string name, int rowIdx)
    {
        int i = rowIdx + 3;   // skip "Row"
        int start = i;
        while (i < name.Length && char.IsDigit(name[i])) i++;
        return i > start ? int.Parse(name.Substring(start, i - start)) : -1;
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
