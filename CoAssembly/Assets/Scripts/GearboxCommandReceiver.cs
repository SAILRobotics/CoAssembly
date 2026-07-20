using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections;
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
///   {"command":"assemble","row":N,"order":[types],"step_delay":s,"slide_seconds":s}
///                                                   → isolate row N and slide its parts into
///                                                     place one type at a time (staggered),
///                                                     in the given type order
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

    [Header("Assembly animation")]
    [Tooltip("Default world-space offset a part starts at before sliding into its final place " +
             "(drops in from above). Used for any part without an override below.")]
    [SerializeField] private Vector3 assembleOffset = new Vector3(0f, 0.20f, 0f);
    [Tooltip("Start-offset overrides so components can come from different directions. Each rule " +
             "matches by any mix of Type / Row / Side (empty Type, Row 0, or Side Any = 'any'). " +
             "E.g. Type=Screw, Row=0, Side=Any => ALL screws come from one direction; " +
             "Type=empty, Side=Left => all left-hand parts. Most specific matching rule wins; " +
             "ties go to the rule higher in the list; unmatched parts use the default offset above.")]
    [SerializeField] private List<AssembleOverride> assembleOverrides = new();
    [Tooltip("Delay between two parts of the SAME type sliding in (e.g. Left then Right gear).")]
    [SerializeField] private float subStaggerSeconds = 0.12f;
    [Tooltip("Fallbacks used when the Python 'assemble' command omits timing.")]
    [SerializeField] private float defaultStepDelay    = 0.35f;
    [SerializeField] private float defaultSlideSeconds  = 0.50f;

    public enum Side { Any, Left, Right }

    [Serializable]
    public struct AssembleOverride
    {
        [Tooltip("Part type to match (e.g. \"Screw\", \"GearRod\"). Empty = ANY type.")]
        public string  type;
        [Tooltip("Row to match (1–4). 0 = ANY row.")]
        public int     row;
        [Tooltip("Side to match. Any = both sides (and side-less parts like GearRod).")]
        public Side    side;
        [Tooltip("World-space offset matching parts start at before sliding home.")]
        public Vector3 offset;
    }

    [Serializable]
    private class GearboxCommand
    {
        public string   command;   // "row"|"toggle"|"show_all"|"reset"|"show_subset"|"ui"|"assemble"
        public int      row;       // "row" | "show_subset" | "ui" | "assemble"
        public string   part;      // "toggle"
        public string[] types;     // "show_subset"
        public bool     show;      // "ui"
        [JsonProperty("checked")]
        public bool     isChecked; // "ui"
        public string[] order;     // "assemble" — part types, in assembly order
        [JsonProperty("step_delay")]
        public float    stepDelay;    // "assemble" — delay between types (0 → use default)
        [JsonProperty("slide_seconds")]
        public float    slideSeconds; // "assemble" — per-part slide duration (0 → use default)
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
        public string               side;    // "Left" / "Right" / "" (side-less, e.g. GearRod)
        public Vector3              restLocalPos;  // authored local position (the slide target)
    }

    private readonly List<PartEntry>                 parts          = new();
    private readonly Dictionary<string, PartEntry>   partsByName    = new();
    private readonly Dictionary<string, PartEntry>   partsByLower   = new();
    private readonly Dictionary<string, PartEntry>   partsByStripped = new();  // underscores removed, lowercased

    // Checkbox tint state (resolved in Start).
    private int                   checkboxColorID;
    private MaterialPropertyBlock checkboxBlock;

    // Assembly-animation state. Bumping the generation invalidates any in-flight slide coroutines.
    private int assembleGen = 0;

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

            // Underscore-agnostic: "Bearing_Row3_Left" and "BearingRow3Left" parse identically.
            string clean  = t.name.Replace("_", "");
            int    rowIdx = clean.IndexOf("Row");
            if (rowIdx < 0) continue;

            int di = rowIdx + 3;                                   // skip "Row"
            while (di < clean.Length && char.IsDigit(clean[di])) di++;
            string side = clean.Substring(di);                    // "Left" / "Right" / ""

            var entry = new PartEntry
            {
                go            = t.gameObject,
                renderer      = r,
                block         = new MaterialPropertyBlock(),
                colorID       = colorID,
                originalColor = colorID != 0 ? mat.GetColor(colorID) : Color.white,
                highlighted   = false,
                type          = clean.Substring(0, rowIdx),   // "GearRod", "Gear", "Bearing", ...
                rowNum        = ParseRow(clean, rowIdx),
                side          = side,
                restLocalPos  = t.localPosition,
            };

            parts.Add(entry);
            partsByName[t.name]                = entry;
            partsByLower[t.name.ToLower()]     = entry;
            partsByStripped[clean.ToLower()]   = entry;
        }
    }

    // Pick the start offset for a part: the most specific matching override (by type/row/side),
    // ties broken by list order; the default offset if nothing matches.
    private Vector3 ResolveOffset(PartEntry p)
    {
        int bestScore = -1;
        Vector3 best = assembleOffset;
        foreach (var o in assembleOverrides)
        {
            if (!string.IsNullOrEmpty(o.type) &&
                !string.Equals(o.type, p.type, StringComparison.OrdinalIgnoreCase)) continue;
            if (o.row != 0 && o.row != p.rowNum) continue;
            if (o.side != Side.Any && !SideMatches(o.side, p.side)) continue;

            int score = (string.IsNullOrEmpty(o.type) ? 0 : 1)
                      + (o.row  != 0        ? 1 : 0)
                      + (o.side != Side.Any ? 1 : 0);
            if (score > bestScore) { bestScore = score; best = o.offset; }
        }
        return best;
    }

    private static bool SideMatches(Side s, string partSide)
    {
        if (s == Side.Left)  return string.Equals(partSide, "Left",  StringComparison.OrdinalIgnoreCase);
        if (s == Side.Right) return string.Equals(partSide, "Right", StringComparison.OrdinalIgnoreCase);
        return true;   // Any
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
                case "assemble":    StartAssemble(cmd.row, cmd.order, cmd.stepDelay, cmd.slideSeconds); break;
                default:
                    Debug.LogWarning($"[GearboxCommandReceiver] Unknown command '{cmd.command}'");
                    break;
            }
        }
    }

    private void ShowOnlyRow(int row)
    {
        CancelAssembly();
        foreach (var p in parts)
            p.go.SetActive(p.rowNum == row);
        Debug.Log($"[GearboxCommandReceiver] 👁 Showing only Row{row}");
    }

    private void ShowAll()
    {
        CancelAssembly();
        foreach (var p in parts)
            p.go.SetActive(true);
        Debug.Log("[GearboxCommandReceiver] 👁 Showing all rows");
    }

    // Show only the given part types of a single row (a "state"); hide everything else.
    private void ShowSubset(int row, string[] types)
    {
        CancelAssembly();
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

        // Centroid of the row's parts at their REST positions — stable even while parts are
        // still mid-slide during an assembly animation.
        Vector3 sum = Vector3.zero;
        int n = 0;
        foreach (var p in parts)
        {
            if (p.rowNum != row) continue;
            sum += p.go.transform.parent != null
                ? p.go.transform.parent.TransformPoint(p.restLocalPos)
                : p.restLocalPos;
            n++;
        }
        Vector3 basePos = (n > 0 ? sum / n : gearboxRoot.position) + uiOffset;

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

    // ── Assembly animation ───────────────────────────────────────────────────
    // Isolate row `row` and slide its parts into place, one type at a time (in `order`),
    // sub-staggering multiple parts of the same type (e.g. Left then Right).
    private void StartAssemble(int row, string[] order, float stepDelay, float slideSeconds)
    {
        CancelAssembly();                       // invalidate any running slide + snap to rest
        int gen = assembleGen;

        float step  = stepDelay    > 0f ? stepDelay    : defaultStepDelay;
        float slide = slideSeconds > 0f ? slideSeconds : defaultSlideSeconds;
        var typeOrder = new List<string>(order ?? Array.Empty<string>());

        // Hide everything; the target parts will be revealed progressively by the coroutine.
        foreach (var p in parts)
            p.go.SetActive(false);

        StartCoroutine(AssembleRoutine(gen, row, typeOrder, step, slide));
    }

    private IEnumerator AssembleRoutine(int gen, int row, List<string> typeOrder,
                                        float stepDelay, float slideSeconds)
    {
        foreach (string type in typeOrder)
        {
            // Parts of this type in this row, sorted by name for a deterministic Left→Right order.
            var group = new List<PartEntry>();
            foreach (var p in parts)
                if (p.rowNum == row && p.type == type)
                    group.Add(p);
            if (group.Count == 0) continue;
            group.Sort((a, b) => string.CompareOrdinal(a.go.name, b.go.name));

            for (int i = 0; i < group.Count; i++)
            {
                if (gen != assembleGen) yield break;   // superseded — stop
                StartCoroutine(SlideIn(gen, group[i], slideSeconds));
                if (i < group.Count - 1)
                    yield return new WaitForSeconds(subStaggerSeconds);
            }
            yield return new WaitForSeconds(stepDelay);
        }
    }

    private IEnumerator SlideIn(int gen, PartEntry p, float duration)
    {
        Transform tr = p.go.transform;
        Vector3 offset = ResolveOffset(p);   // per-part / per-type direction, else the default

        // Position at the start pose BEFORE activating, so there's no one-frame flash at rest.
        Vector3 rest0 = tr.parent != null ? tr.parent.TransformPoint(p.restLocalPos) : p.restLocalPos;
        tr.position = rest0 + offset;
        p.go.SetActive(true);

        float t = 0f;
        while (t < duration)
        {
            if (gen != assembleGen)          // superseded: snap home and bail
            {
                tr.localPosition = p.restLocalPos;
                yield break;
            }
            t += Time.deltaTime;
            float k = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / duration));

            // Recompute each frame so the slide stays correct even if the gearbox root moves.
            Vector3 restWorld = tr.parent != null ? tr.parent.TransformPoint(p.restLocalPos)
                                                  : p.restLocalPos;
            Vector3 startWorld = restWorld + offset;
            tr.position = Vector3.Lerp(startWorld, restWorld, k);
            yield return null;
        }
        tr.localPosition = p.restLocalPos;   // land exactly on the authored pose
    }

    // Invalidate any in-flight slide coroutines and restore every part to its rest position.
    private void CancelAssembly()
    {
        assembleGen++;
        foreach (var p in parts)
            p.go.transform.localPosition = p.restLocalPos;
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

        // Match exact, then case-insensitive, then underscore-agnostic — so "BearingRow3Left"
        // and "Bearing_Row3_Left" both resolve regardless of how the part is named.
        if (!partsByName.TryGetValue(name, out PartEntry entry)
            && !partsByLower.TryGetValue(name.ToLower(), out entry))
            partsByStripped.TryGetValue(name.Replace("_", "").ToLower(), out entry);

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
