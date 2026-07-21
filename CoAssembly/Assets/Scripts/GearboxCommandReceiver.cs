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
/// Commands (JSON, sent by gearbox_control.py):
///   {"command":"row","row":N}                       → show only parts of row N
///   {"command":"toggle","part":".."}                → flip a single part's highlight color
///   {"command":"show_all"} / {"command":"reset"}    → show every part / reset all colors
///   {"command":"ui","show":bool,"row":N,"checked":bool,"blocked":bool}
///                                                   → place + tint the checkbox/X (row 0 = whole
///                                                     gearbox); red if blocked, green if checked
///   {"command":"stage","row":R,"stage":N,"done_stages":[..],"step_delay":s,"slide_seconds":s}
///                                                   → play the 7-stage assembly: show stages 1..N
///                                                     of row R with the staging→seat animation;
///                                                     green the parts whose stage is in done_stages
///   {"command":"recolor","row":R,"stage":S,"done":bool}
///                                                   → green / un-green the parts that seat at stage
///                                                     S of row R (no motion) — fired on check/uncheck
///
/// The 7-stage model + two-position (staging→seat) motion + green-on-complete live here (Unity owns
/// the choreography); Python (gearbox_control.py) owns the dependency/lock logic. Mirrors the
/// receiver pattern in ToolColorReceiver.cs: bind a SubscriberSocket, receive JSON on a background
/// thread into a ConcurrentQueue, apply on the main thread in Update().
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
    [Tooltip("Fallbacks used when the Python command omits timing.")]
    [SerializeField] private float defaultStepDelay    = 0.35f;
    [SerializeField] private float defaultSlideSeconds  = 0.50f;
    [Tooltip("World-space offset a sub-assembly floats at (above the board) while it is being " +
             "built, before it drops onto the board when its stage seats. Tune to your scene scale.")]
    [SerializeField] private Vector3 stagingOffset = new Vector3(0f, 0.15f, 0f);

    [Header("Stage completion colors")]
    [Tooltip("Color a part turns when its stage is checked complete (\"that portion is done\").")]
    [SerializeField] private Color seatedColor  = new Color(0.15f, 0.80f, 0.20f, 1f);  // green
    [Tooltip("Color the checkbox tints when the current stage is locked (prerequisites unmet).")]
    [SerializeField] private Color blockedColor = new Color(0.85f, 0.15f, 0.15f, 1f);  // red

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
        public float    stepDelay;    // "stage"/"assemble" — delay between groups (0 → use default)
        [JsonProperty("slide_seconds")]
        public float    slideSeconds; // "stage"/"assemble" — per-part slide duration (0 → default)
        public int      stage;        // "stage"/"recolor" — 1..7
        [JsonProperty("done_stages")]
        public int[]    doneStages;   // "stage" — stages already completed for this row (→ green)
        public bool     done;         // "recolor" — completed (green) or un-done (original)
        public bool     blocked;      // "ui" — current stage locked → red checkbox
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
        public int                  rowNum;  // parsed row number; 0 = shared (BaseBoard)
        public string               side;    // "Left" / "Right" / "" (side-less, e.g. GearRod)
        public Vector3              restLocalPos;  // authored local (FINAL) position
        public int                  appearStage;   // first stage the part is shown (staging or final)
        public int                  seatStage;     // stage the part moves STAGING → FINAL, then fixed
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
        // A "part" = a mesh transform whose name contains "Row" ({Type}Row{N}{Side}), plus the
        // shared "BaseBoard" mesh (no "Row"). Non-part nodes (incl. the root) are excluded, so we
        // never SetActive(false) the GameObject this script lives on.
        foreach (Transform t in gearboxRoot.GetComponentsInChildren<Transform>(true))
        {
            string clean = t.name.Replace("_", "");
            int    rowIdx = clean.IndexOf("Row");
            bool   isRow  = rowIdx >= 0;
            bool   isBase = !isRow && clean.IndexOf("Base", StringComparison.OrdinalIgnoreCase) >= 0;
            if (!isRow && !isBase) continue;

            Renderer r = t.GetComponent<Renderer>();
            if (r == null) continue;
            Material mat = r.sharedMaterial;
            if (mat == null) continue;

            int colorID = ResolveColorID(mat);
            if (colorID == 0)
                Debug.LogWarning($"[GearboxCommandReceiver] '{t.name}' has no known base-color " +
                                 $"property (shader '{mat.shader.name}'); color disabled for it.");

            string type, side;
            int    rowNum;
            if (isRow)
            {
                int di = rowIdx + 3;                                   // skip "Row"
                while (di < clean.Length && char.IsDigit(clean[di])) di++;
                side   = clean.Substring(di);                          // "Left" / "Right" / ""
                type   = clean.Substring(0, rowIdx);                  // "GearRod", "Bearing", ...
                rowNum = ParseRow(clean, rowIdx);
            }
            else                                                       // BaseBoard — shared
            {
                type = "BaseBoard"; side = ""; rowNum = 0;
            }

            StagesOf(type, side, out int appear, out int seat);

            var entry = new PartEntry
            {
                go            = t.gameObject,
                renderer      = r,
                block         = new MaterialPropertyBlock(),
                colorID       = colorID,
                originalColor = colorID != 0 ? mat.GetColor(colorID) : Color.white,
                highlighted   = false,
                type          = type,
                rowNum        = rowNum,
                side          = side,
                restLocalPos  = t.localPosition,
                appearStage   = appear,
                seatStage     = seat,
            };

            parts.Add(entry);
            partsByName[t.name]                = entry;
            partsByLower[t.name.ToLower()]     = entry;
            partsByStripped[clean.ToLower()]   = entry;
        }
    }

    // The 7-stage assembly model: when each part first appears (at STAGING for sub-assemblies, or
    // straight to FINAL for board-level parts) and when it seats (STAGING → FINAL, then fixed).
    private static void StagesOf(string type, string side, out int appear, out int seat)
    {
        bool left = string.Equals(side, "Left", StringComparison.OrdinalIgnoreCase);
        switch (type.ToLowerInvariant())
        {
            case "bearing": case "stand":  appear = left ? 1 : 3; seat = left ? 4 : 5; break;
            case "gearrod": case "gear": case "pin":  appear = 2; seat = 4;            break;
            case "screw":                  appear = left ? 4 : 5; seat = left ? 4 : 5; break;
            case "crankhandle":            appear = 6; seat = 6;                        break;
            case "baseboard":              appear = 4; seat = 4;                        break;
            default:                       appear = 0; seat = 0;                        break;
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
                case "ui":          ShowUi(cmd.row, cmd.show, cmd.isChecked, cmd.blocked); break;
                case "assemble":    StartAssemble(cmd.row, cmd.order, cmd.stepDelay, cmd.slideSeconds); break;
                case "stage":       PlayStage(cmd.row, cmd.stage, cmd.doneStages, cmd.stepDelay, cmd.slideSeconds); break;
                case "recolor":     Recolor(cmd.row, cmd.stage, cmd.done); break;
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

    // Show/hide + place the checkbox and reset-X near the active row (row 0 = whole gearbox), and
    // tint the checkbox: red if blocked, else green if checked, else grey.
    private void ShowUi(int row, bool show, bool isChecked, bool blocked)
    {
        if (!show)
        {
            if (checkboxObject) checkboxObject.gameObject.SetActive(false);
            if (resetObject)    resetObject.gameObject.SetActive(false);
            return;
        }

        // Centroid of the row's parts at their REST positions (all parts when row==0) — stable
        // even while parts are still mid-slide during an assembly animation.
        Vector3 sum = Vector3.zero;
        int n = 0;
        foreach (var p in parts)
        {
            if (row != 0 && p.rowNum != row) continue;
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
        ApplyCheckboxVisual(isChecked, blocked);
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

    private void ApplyCheckboxVisual(bool isChecked, bool blocked)
    {
        if (checkboxRenderer == null || checkboxColorID == 0 || checkboxBlock == null) return;
        Color c = blocked ? blockedColor : (isChecked ? checkedColor : uncheckedColor);
        checkboxRenderer.GetPropertyBlock(checkboxBlock);
        checkboxBlock.SetColor(checkboxColorID, c);
        checkboxRenderer.SetPropertyBlock(checkboxBlock);
    }

    // ── Stage choreography (7-stage dependency assembly) ─────────────────────
    // Which appear-stages are shown when displaying stage N: N itself plus its prerequisite closure.
    // Stages 1/2/3 are INDEPENDENT (shown alone); 4 depends on 1&2; 5 on 3&4; 6 on 5; 7 shows all.
    // So opening stage 4 reveals stages 1,2,4 but NOT 3 (per the dependency graph).
    private static HashSet<int> VisibleAppearStages(int n)
    {
        switch (n)
        {
            case 1:  return new HashSet<int> { 1 };
            case 2:  return new HashSet<int> { 2 };
            case 3:  return new HashSet<int> { 3 };
            case 4:  return new HashSet<int> { 1, 2, 4 };
            case 5:  return new HashSet<int> { 1, 2, 3, 4, 5 };
            case 6:  return new HashSet<int> { 1, 2, 3, 4, 5, 6 };
            default: return new HashSet<int> { 1, 2, 3, 4, 5, 6 };   // 7 (verify): everything
        }
    }

    // Show stage N of row R (row 0 = whole gearbox). Parts whose appear-stage is not in stage N's
    // dependency closure are hidden; parts already at their pose are shown static; this stage's
    // motion is played by StageRoutine. `doneStages` are the row's completed stages → green.
    private void PlayStage(int row, int n, int[] doneStages, float stepDelay, float slideSeconds)
    {
        int gen = ++assembleGen;   // supersede any running animation (positions set explicitly below)
        float step  = stepDelay    > 0f ? stepDelay    : defaultStepDelay;
        float slide = slideSeconds > 0f ? slideSeconds : defaultSlideSeconds;
        var done    = new HashSet<int>(doneStages ?? Array.Empty<int>());
        var visible = VisibleAppearStages(n);

        foreach (var p in parts)
        {
            if (!InView(p, row) || !visible.Contains(p.appearStage)) { p.go.SetActive(false); continue; }
            if (p.appearStage == n || p.seatStage == n)
            {
                // Animated by StageRoutine. Pre-place at its START pose NOW so it is never seen at
                // its final spot first (which caused the "sit at rest → jump → slide" glitch). A
                // part that merely drops this stage waits VISIBLY at STAGING (it was built earlier);
                // a first-appearing part stays hidden until its group reveals it at the start offset.
                p.go.transform.localPosition = StageStartLocal(p, n);
                p.go.SetActive(p.appearStage < n);
                continue;
            }
            // Static: place at its current pose (FINAL if seated by now, else STAGING) and color it.
            p.go.transform.localPosition = (p.seatStage <= n) ? p.restLocalPos : StagingLocal(p);
            p.go.SetActive(true);
            ApplyPartColor(p, done);
        }

        StartCoroutine(StageRoutine(gen, row, n, done, step, slide));
        Debug.Log($"[GearboxCommandReceiver] ▶ stage row{row} stage{n}");
    }

    private IEnumerator StageRoutine(int gen, int row, int n, HashSet<int> done,
                                     float stepDelay, float slideSeconds)
    {
        foreach (string[] group in GroupsForStage(n))
        {
            // Parts of this row that this group animates this stage (appear==n or seat==n).
            var members = new List<PartEntry>();
            foreach (var p in parts)
            {
                if (!InView(p, row)) continue;
                if (p.appearStage != n && p.seatStage != n) continue;
                foreach (string sel in group)
                    if (MatchesSelector(sel, p)) { members.Add(p); break; }
            }
            if (members.Count == 0) continue;
            members.Sort((a, b) => string.CompareOrdinal(a.go.name, b.go.name));

            // A "seat" group (parts dropping STAGING→FINAL) moves together; a first-appear group
            // sub-staggers so it reads piece by piece.
            bool together = members.TrueForAll(m => m.seatStage == n && m.appearStage < n);
            for (int i = 0; i < members.Count; i++)
            {
                if (gen != assembleGen) yield break;
                StartCoroutine(SlideLocal(gen, members[i], n, done, slideSeconds));
                if (!together && i < members.Count - 1)
                    yield return new WaitForSeconds(subStaggerSeconds);
            }
            yield return new WaitForSeconds(stepDelay);
        }
    }

    // The local pose a part starts its stage-n motion from. A first-appearing part (appear==n)
    // starts offset AWAY from its end pose (which is STAGING while un-seated, else FINAL); a part
    // that only drops this stage (seat==n, appear<n) starts at STAGING, where it was built earlier.
    private Vector3 StageStartLocal(PartEntry p, int n)
    {
        Vector3 stagingLocal = StagingLocal(p);
        Vector3 endLocal = (p.seatStage <= n) ? p.restLocalPos : stagingLocal;
        if (p.appearStage == n)
        {
            Transform par = p.go.transform.parent;
            Vector3 offsetLocal = par != null ? par.InverseTransformVector(ResolveOffset(p)) : ResolveOffset(p);
            return endLocal + offsetLocal;
        }
        return stagingLocal;
    }

    // Animate one part into its stage-n pose. appear==n: slide in (from the ResolveOffset direction)
    // to STAGING (if still un-seated) or FINAL; seat==n (drop): move STAGING → FINAL.
    private IEnumerator SlideLocal(int gen, PartEntry p, int n, HashSet<int> done, float duration)
    {
        Transform tr = p.go.transform;
        Vector3 endLocal   = (p.seatStage <= n) ? p.restLocalPos : StagingLocal(p);
        Vector3 startLocal = StageStartLocal(p, n);

        tr.localPosition = startLocal;
        p.go.SetActive(true);
        ApplyColor(p, p.originalColor);   // uncolored while it moves

        float t = 0f;
        while (t < duration)
        {
            if (gen != assembleGen) yield break;   // superseded — the new stage owns positions
            t += Time.deltaTime;
            float k = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / duration));
            tr.localPosition = Vector3.Lerp(startLocal, endLocal, k);
            yield return null;
        }
        tr.localPosition = endLocal;
        ApplyPartColor(p, done);          // green if this part's stage is already complete
    }

    // Per-stage ordered animation groups (each inner array is a group of "Type:Side" selectors;
    // groups play in order with step_delay between them). Mirrors the plan's choreography.
    private static IEnumerable<string[]> GroupsForStage(int n)
    {
        switch (n)
        {
            case 1: return new[] { new[] { "Stand:Left",  "Bearing:Left"  } };
            case 2: return new[] { new[] { "GearRod:Any" }, new[] { "Gear:Any" }, new[] { "Pin:Any" } };
            case 3: return new[] { new[] { "Stand:Right", "Bearing:Right" } };
            case 4: return new[] { new[] { "BaseBoard:Any" },
                                   new[] { "Bearing:Left", "Stand:Left" },
                                   new[] { "Screw:Left" },
                                   new[] { "GearRod:Any", "Gear:Any", "Pin:Any" } };
            case 5: return new[] { new[] { "Bearing:Right", "Stand:Right" }, new[] { "Screw:Right" } };
            case 6: return new[] { new[] { "CrankHandle:Any" } };
            default: return Array.Empty<string[]>();   // stage 7 (and any other): nothing animates
        }
    }

    private static bool MatchesSelector(string selector, PartEntry p)
    {
        int c = selector.IndexOf(':');
        string t    = c >= 0 ? selector.Substring(0, c) : selector;
        string side = c >= 0 ? selector.Substring(c + 1) : "Any";
        if (!string.Equals(t, p.type, StringComparison.OrdinalIgnoreCase)) return false;
        if (string.Equals(side, "Any", StringComparison.OrdinalIgnoreCase)) return true;
        return string.Equals(side, p.side, StringComparison.OrdinalIgnoreCase);
    }

    // A part is in the current view if it belongs to the row, or is shared (BaseBoard, rowNum 0);
    // row 0 (stage 7) shows everything.
    private bool InView(PartEntry p, int row) =>
        row == 0 || p.rowNum == row || p.rowNum == 0;

    private Vector3 StagingLocal(PartEntry p)
    {
        Transform par = p.go.transform.parent;
        Vector3 offLocal = par != null ? par.InverseTransformVector(stagingOffset) : stagingOffset;
        return p.restLocalPos + offLocal;
    }

    // Green a part iff its seat stage is complete (BaseBoard stays neutral). Used for static parts
    // in PlayStage and at the end of an animated part's move.
    private void ApplyPartColor(PartEntry p, HashSet<int> doneStages)
    {
        if (p.colorID == 0 || p.type == "BaseBoard") return;
        ApplyColor(p, doneStages.Contains(p.seatStage) ? seatedColor : p.originalColor);
    }

    // Tint every part that seats at `stage` in `row` green (done) or original (un-done). No motion.
    private void Recolor(int row, int stage, bool done)
    {
        foreach (var p in parts)
        {
            if (p.rowNum != row || p.seatStage != stage || p.type == "BaseBoard" || p.colorID == 0)
                continue;
            ApplyColor(p, done ? seatedColor : p.originalColor);
        }
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

    // Full color reset: restore every part to its original color (clears both the yellow toggle
    // highlights and the green "done" seating colors).
    private void ResetHighlights()
    {
        foreach (var p in parts)
        {
            p.highlighted = false;
            if (p.colorID != 0) ApplyColor(p, p.originalColor);
        }
        Debug.Log("[GearboxCommandReceiver] ♻ Reset all part colors");
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
