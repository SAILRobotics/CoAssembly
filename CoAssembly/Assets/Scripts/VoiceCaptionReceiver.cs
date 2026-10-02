using System;
using System.Collections.Concurrent;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Shows the voice assistant's conversation in AR: what it heard ("You"), what it
/// replied ("Assistant"), and a status dot (listening / hearing / transcribing /
/// thinking / speaking). Python (task_graph/voice_caption.py) publishes on port 5028:
///   {"kind":"status","status":"listening","level":0.2}
///   {"kind":"heard","text":"get me the hex bit"}
///   {"kind":"reply","text":"Which one: H5 or H3?","awaiting_answer":true}
///   {"kind":"notice","text":"Didn't catch that — I was speaking."}
///
/// Setup: add to any empty GameObject in the scene (NOT under WorldRoot). The panel
/// is built at runtime as a world-space canvas and lazily follows the head: it sits
/// below eye level and only re-centres when you look well away, so it never covers
/// the work area. Lines fade after a few seconds; a reply that asks a question stays
/// until you answer.
/// </summary>
public class VoiceCaptionReceiver : MonoBehaviour
{
    [Header("NetMQ")]
    [SerializeField] private int port = 5028;

    [Header("Placement")]
    [Tooltip("Head transform to follow; defaults to Camera.main (CenterEyeAnchor).")]
    [SerializeField] private Transform head;
    [SerializeField] private float distance = 0.9f;
    [SerializeField] private float belowEyeDegrees = 18f;
    [Tooltip("Re-centre only when the panel is this far (deg) from where you look.")]
    [SerializeField] private float recenterAngle = 30f;
    [SerializeField] private float followSmoothTime = 0.35f;

    [Header("Timing")]
    [SerializeField] private float lineHoldSeconds = 8f;
    [SerializeField] private float fadeSeconds = 1f;
    [SerializeField] private float noticeSeconds = 3f;
    [Tooltip("Show 'Voice offline' when no status arrives for this long.")]
    [SerializeField] private float offlineSeconds = 5f;

    [Serializable]
    private class Message
    {
        public string kind;
        public string status;
        public float level;
        public string text;
        public bool awaiting_answer;
    }

    private SubscriberSocket _sock;
    private Thread _thread;
    private volatile bool _running;
    private readonly ConcurrentQueue<Message> _queue = new();

    // UI
    private Transform _panel;
    private Image _dot;
    private RectTransform _levelBar;
    private TextMeshProUGUI _statusText;
    private TextMeshProUGUI _heardText;
    private TextMeshProUGUI _replyText;

    // State
    private string _status = "loading";
    private float _level;
    private float _lastStatusTime = -999f;
    private float _heardTime = -999f;
    private float _replyTime = -999f;
    private bool _awaitingAnswer;
    private string _notice;
    private float _noticeTime = -999f;
    private Vector3 _followVelocity;
    private bool _placed;

    private static readonly Color Grey   = new(0.65f, 0.65f, 0.65f);
    private static readonly Color Green  = new(0.30f, 0.90f, 0.40f);
    private static readonly Color Yellow = new(1.00f, 0.85f, 0.25f);
    private static readonly Color Orange = new(1.00f, 0.60f, 0.20f);
    private static readonly Color Cyan   = new(0.30f, 0.85f, 1.00f);
    private static readonly Color Red    = new(1.00f, 0.30f, 0.30f);

    private void Start()
    {
        if (head == null && Camera.main != null) head = Camera.main.transform;
        BuildPanel();

        AsyncIO.ForceDotNet.Force();
        _sock = new SubscriberSocket();
        _sock.Bind($"tcp://0.0.0.0:{port}");
        _sock.Subscribe("");
        NetMQManager.RegisterReceiver();

        _running = true;
        _thread = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
        Debug.Log($"[VoiceCaption] 📡 SUB bound on tcp://0.0.0.0:{port}");
    }

    private void ReceiveLoop()
    {
        while (_running)
        {
            try
            {
                if (_sock.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string msg))
                {
                    var m = JsonConvert.DeserializeObject<Message>(msg);
                    if (m != null) _queue.Enqueue(m);
                }
            }
            catch (TerminatingException) { break; }
            catch (ObjectDisposedException) { break; }
            catch (Exception e) { if (_running) Debug.LogWarning($"[VoiceCaption] {e.Message}"); }
        }
    }

    private void Update()
    {
        float now = Time.time;
        while (_queue.TryDequeue(out var m)) Apply(m, now);

        RefreshStatus(now);
        RefreshLine(_heardText, _heardTime, now, keep: false);
        RefreshLine(_replyText, _replyTime, now, keep: _awaitingAnswer);
        Follow();
    }

    private void Apply(Message m, float now)
    {
        switch (m.kind)
        {
            case "status":
                _status = m.status ?? _status;
                _level = Mathf.Clamp01(m.level);
                _lastStatusTime = now;
                break;
            case "heard":
                _heardText.text = $"<color=#9A9A9A>You:</color> {m.text}";
                _heardTime = now;
                // Answering a question releases the reply to fade normally.
                if (_awaitingAnswer) { _awaitingAnswer = false; _replyTime = now; }
                break;
            case "reply":
                _replyText.text = $"<color=#4DD9FF>Assistant:</color> {m.text}";
                _replyTime = now;
                _awaitingAnswer = m.awaiting_answer;
                break;
            case "notice":
                _notice = m.text;
                _noticeTime = now;
                break;
        }
    }

    private void RefreshStatus(float now)
    {
        string label;
        Color color;
        bool offline = now - _lastStatusTime > offlineSeconds;
        if (offline) { label = "Voice offline"; color = Grey; }
        else
        {
            switch (_status)
            {
                case "listening":    label = "Listening";               color = Green;  break;
                case "hearing":      label = "Hearing you…";            color = Green;  break;
                case "transcribing": label = "Transcribing…";           color = Yellow; break;
                case "thinking":     label = "Thinking…";               color = Orange; break;
                case "speaking":     label = "Speaking — not listening"; color = Cyan;  break;
                case "muted":        label = "Voice off";               color = Grey;   break;
                case "error":        label = "Voice error";             color = Red;    break;
                default:             label = "Starting voice…";         color = Grey;   break;
            }
        }

        if (_notice != null && now - _noticeTime < noticeSeconds)
        {
            label = _notice;
            _statusText.color = Grey;
        }
        else
        {
            _statusText.color = Color.white;
        }
        _statusText.text = label;

        // Dot pulses while working; level bar follows the mic while listening.
        bool busy = !offline && (_status == "transcribing" || _status == "thinking");
        float pulse = busy ? 0.55f + 0.45f * Mathf.Abs(Mathf.Sin(now * 4f)) : 1f;
        _dot.color = new Color(color.r, color.g, color.b, pulse);
        bool showLevel = !offline && (_status == "listening" || _status == "hearing");
        _levelBar.gameObject.SetActive(showLevel);
        _levelBar.sizeDelta = new Vector2(Mathf.Lerp(4f, 120f, _level), 6f);
    }

    private void RefreshLine(TextMeshProUGUI line, float shownAt, float now, bool keep)
    {
        float age = now - shownAt;
        float alpha = keep || age < lineHoldSeconds
            ? 1f
            : 1f - Mathf.Clamp01((age - lineHoldSeconds) / fadeSeconds);
        bool visible = alpha > 0.01f && !string.IsNullOrEmpty(line.text);
        if (line.gameObject.activeSelf != visible) line.gameObject.SetActive(visible);
        if (visible) line.alpha = alpha;
    }

    private void Follow()
    {
        if (head == null)
        {
            if (Camera.main == null) return;
            head = Camera.main.transform;
        }
        Vector3 forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        if (forward.sqrMagnitude < 1e-4f) forward = head.up;   // looking straight down/up
        forward.Normalize();
        Vector3 right = Vector3.Cross(Vector3.up, forward);
        Vector3 dir = Quaternion.AngleAxis(belowEyeDegrees, right) * forward;
        Vector3 target = head.position + dir * distance;

        Vector3 toPanel = Vector3.ProjectOnPlane(_panel.position - head.position, Vector3.up);
        bool farOff = !_placed
                      || Vector3.Angle(forward, toPanel) > recenterAngle
                      || Mathf.Abs((_panel.position - head.position).magnitude - distance) > 0.3f;
        if (!_placed)
        {
            _panel.position = target;
            _placed = true;
        }
        else if (farOff || _followVelocity.sqrMagnitude > 1e-6f)
        {
            _panel.position = Vector3.SmoothDamp(
                _panel.position, target, ref _followVelocity, followSmoothTime);
            if ((_panel.position - target).sqrMagnitude < 1e-4f) _followVelocity = Vector3.zero;
        }
        _panel.rotation = Quaternion.LookRotation(_panel.position - head.position, Vector3.up);
    }

    // ── Runtime UI construction ─────────────────────────────────────────────────

    private void BuildPanel()
    {
        var canvasGo = new GameObject("VoiceCaptionPanel",
            typeof(Canvas), typeof(CanvasScaler), typeof(VerticalLayoutGroup),
            typeof(ContentSizeFitter), typeof(Image));
        _panel = canvasGo.transform;
        _panel.SetParent(transform, false);
        _panel.localScale = Vector3.one * 0.001f;   // 1 UI unit = 1 mm

        var canvas = canvasGo.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.sortingOrder = 100;
        var rect = (RectTransform)_panel;
        rect.sizeDelta = new Vector2(420f, 100f);

        canvasGo.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.6f);
        var layout = canvasGo.GetComponent<VerticalLayoutGroup>();
        layout.padding = new RectOffset(16, 16, 10, 12);
        layout.spacing = 6f;
        layout.childControlWidth = true;
        layout.childControlHeight = true;
        layout.childForceExpandWidth = true;
        layout.childForceExpandHeight = false;
        var fitter = canvasGo.GetComponent<ContentSizeFitter>();
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        // Status row: dot + label + mic level bar.
        var row = new GameObject("Status", typeof(RectTransform), typeof(HorizontalLayoutGroup));
        row.transform.SetParent(_panel, false);
        var rowLayout = row.GetComponent<HorizontalLayoutGroup>();
        rowLayout.spacing = 8f;
        rowLayout.childAlignment = TextAnchor.MiddleLeft;
        rowLayout.childControlWidth = false;
        rowLayout.childControlHeight = false;
        rowLayout.childForceExpandWidth = false;

        _dot = NewImage("Dot", row.transform, new Vector2(14f, 14f), Grey);
        _statusText = NewText("StatusText", row.transform, 18f, FontStyles.Bold);
        ((RectTransform)_statusText.transform).sizeDelta = new Vector2(250f, 24f);
        _levelBar = (RectTransform)NewImage("Level", row.transform, new Vector2(4f, 6f), Green).transform;

        _heardText = NewText("Heard", _panel, 20f, FontStyles.Normal);
        _replyText = NewText("Reply", _panel, 22f, FontStyles.Normal);
        _heardText.text = "";
        _replyText.text = "";
        _heardText.gameObject.SetActive(false);
        _replyText.gameObject.SetActive(false);
    }

    private static Image NewImage(string name, Transform parent, Vector2 size, Color color)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(Image));
        go.transform.SetParent(parent, false);
        ((RectTransform)go.transform).sizeDelta = size;
        var image = go.GetComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
        return image;
    }

    private static TextMeshProUGUI NewText(string name, Transform parent, float size, FontStyles style)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(TextMeshProUGUI));
        go.transform.SetParent(parent, false);
        var text = go.GetComponent<TextMeshProUGUI>();
        text.fontSize = size;
        text.fontStyle = style;
        text.color = Color.white;
        text.enableWordWrapping = true;
        text.raycastTarget = false;
        return text;
    }

    private void OnDestroy()
    {
        _running = false;
        if (_thread != null && _thread.IsAlive) _thread.Join(500);
        if (_sock != null)
        {
            try { _sock.Close(); _sock.Dispose(); } catch { }
            _sock = null;
            NetMQManager.UnregisterReceiver();
        }
    }
}
