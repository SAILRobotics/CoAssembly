using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Barebones receiver for the workholding AR-box test harness (workholding_testing.py).
///
/// Binds WORKHOLDING_BOX_PORT (5026) — its OWN port, distinct from GripStateReceiver's 5012 so the
/// two never clash — and consumes the SAME message schema as GripStateReceiver
/// (grip_state, box_pos, box_rot_xyzw, box_size), but — unlike GripStateReceiver, which only
/// reveals the box on an ISDK grab — this ALWAYS shows the box at the received pose. That lets
/// the harness park the box at each of its 10 test poses with no grabbing involved.
///
/// Inspector setup:
///   arBox     — the board model used as the target-board template
///   worldRoot — WorldRoot transform (same object WorldRoot.cs / ToolSpawner drive)
///
/// Mirrors the socket / background-thread / ConcurrentQueue / Update lifecycle and the
/// NetMQManager register/unregister pattern from GripStateReceiver.cs.
/// </summary>
public class WorkholdingBoxReceiver : MonoBehaviour
{
    [Header("AR box")]
    public GameObject arBox;
    public bool applyIncomingBoxSize = true;
    [Tooltip("Clone arBox so the target board remains independent from the board attached to the robot.")]
    public bool cloneTargetBoard = true;

    [Header("Board appearance")]
    public bool overrideBoardMaterial = true;
    [Range(0f, 1f)] public float boardAlpha = 0.45f;
    public Color boardTint = Color.black;
    public Color farColor = Color.red;
    public Color nearColor = new Color(1f, 0.42f, 0.02f, 1f);
    public Color reachedColor = Color.green;
    public Color blackColor = Color.black;

    [Header("Target gripper")]
    [Tooltip("Visual template only. A non-interactive ghost clone is created at runtime; this object is never moved, hidden, or recolored.")]
    public GameObject targetGripper;
    public bool showTargetGripper = true;
    public float targetGripperForwardOffset = 0.23215f;
    public bool overrideTargetGripperMaterial = true;
    [Range(0f, 1f)] public float targetGripperAlpha = 0.18f;
    public Color targetGripperTint = new Color(0.70f, 0.84f, 1.00f, 1f);

    [Header("Parent")]
    public Transform worldRoot;

    [Header("NetMQ")]
    [SerializeField] private int port = 5026;   // WORKHOLDING_BOX_PORT (not 5012 / GRIP_STATE_PORT)

    [Serializable]
    private class BoxMessage
    {
        public string  grip_state;      // far / near / reached proximity color
        public float[] box_pos;
        public float[] box_rot_xyzw;
        public float[] box_size;
        public float[] box_color;
        public float[] gripper_color;
    }

    private class PoseData
    {
        public string     proximityState;
        public Vector3    pos;
        public Quaternion rot;
        public Vector3    size;
        public Color?     color;
        public Color?     gripperColor;
    }

    private SubscriberSocket _socket;
    private Thread           _thread;
    private volatile bool    _running;
    private readonly ConcurrentQueue<PoseData> _queue = new();
    private Renderer[]       _renderers;
    private Material         _boardMaterial;
    private Renderer[]       _targetGripperRenderers;
    private GameObject       _targetGripperVisual;
    private Material         _targetGripperMaterial;
    private MaterialPropertyBlock _targetGripperColorBlock;
    private bool             _loggedShow;
    private string           _lastProximityState = "";

    private void Start()
    {
        // WorkholdingTesting also assigns its robot-attached board to GripStateReceiver.
        // That receiver deliberately deactivates the attached board in idle states.  A
        // separate target instance prevents it from hiding or moving this desired pose.
        if (arBox != null && cloneTargetBoard)
        {
            Transform parent = worldRoot != null ? worldRoot : arBox.transform.parent;
            arBox = Instantiate(arBox, parent, false);
            arBox.name = "TargetHalfBoard";
            arBox.SetActive(true);
        }

        // Hide until the first pose arrives by toggling the RENDERER(s), not the GameObject:
        // this component may live on arBox itself, and SetActive(false) on our own GameObject
        // would stop Update() from ever running (so the box could never be shown again).
        if (arBox != null)
        {
            _renderers = arBox.GetComponentsInChildren<Renderer>(true);
            ApplyBoardMaterial();
            SetRenderersVisible(_renderers, false);
        }

        if (targetGripper != null)
        {
            Transform parent = worldRoot != null
                ? worldRoot : targetGripper.transform.parent;
            _targetGripperVisual = Instantiate(targetGripper, parent, false);
            _targetGripperVisual.name = "TargetGripperGhost";

            // The target is a visual guide only. In particular, cloned tool-id 200
            // publishers must not be able to toggle Hybrid freedrive.
            foreach (MonoBehaviour behaviour in
                     _targetGripperVisual.GetComponentsInChildren<MonoBehaviour>(true))
                behaviour.enabled = false;
            foreach (Collider collider in
                     _targetGripperVisual.GetComponentsInChildren<Collider>(true))
                collider.enabled = false;
            foreach (Rigidbody body in
                     _targetGripperVisual.GetComponentsInChildren<Rigidbody>(true))
            {
                body.isKinematic = true;
                body.detectCollisions = false;
            }

            _targetGripperRenderers =
                _targetGripperVisual.GetComponentsInChildren<Renderer>(true);
            _targetGripperColorBlock = new MaterialPropertyBlock();
            ApplyTargetGripperMaterial();
            SetRenderersVisible(_targetGripperRenderers, false);
        }

        NetMQManager.RegisterReceiver();
        _running = true;
        _thread  = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
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
                Debug.Log($"[WorkholdingBoxReceiver] SUB bound on tcp://0.0.0.0:{port}");
                while (_running)
                {
                    try
                    {
                        if (_socket.TryReceiveFrameString(
                                TimeSpan.FromMilliseconds(100), out string msg))
                        {
                            var d = JsonConvert.DeserializeObject<BoxMessage>(msg);
                            if (d == null || d.box_pos == null || d.box_rot_xyzw == null
                                || d.box_size == null) continue;
                            Color? incomingColor = null;
                            if (d.box_color != null && d.box_color.Length >= 3)
                            {
                                float a = d.box_color.Length >= 4 ? d.box_color[3] : boardAlpha;
                                incomingColor = new Color(
                                    d.box_color[0], d.box_color[1], d.box_color[2], a);
                            }
                            Color? incomingGripperColor = null;
                            if (d.gripper_color != null && d.gripper_color.Length >= 3)
                            {
                                float a = d.gripper_color.Length >= 4
                                    ? d.gripper_color[3] : targetGripperAlpha;
                                incomingGripperColor = new Color(
                                    d.gripper_color[0], d.gripper_color[1],
                                    d.gripper_color[2], a);
                            }
                            _queue.Enqueue(new PoseData
                            {
                                proximityState = d.grip_state,
                                pos  = new Vector3(d.box_pos[0], d.box_pos[1], d.box_pos[2]),
                                rot  = new Quaternion(d.box_rot_xyzw[0], d.box_rot_xyzw[1],
                                                      d.box_rot_xyzw[2], d.box_rot_xyzw[3]),
                                size = new Vector3(d.box_size[0], d.box_size[1], d.box_size[2]),
                                color = incomingColor,
                                gripperColor = incomingGripperColor,
                            });
                        }
                    }
                    catch (TerminatingException)    { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e) { if (_running) Debug.LogWarning("[WorkholdingBoxReceiver] " + e.Message); }
                }
            }
        }
        catch (Exception e) { if (_running) Debug.LogWarning("[WorkholdingBoxReceiver] Outer: " + e.Message); }
    }

    private void Update()
    {
        PoseData latest = null;
        while (_queue.TryDequeue(out var d)) latest = d;
        if (latest == null || arBox == null) return;

        // The pose arrives in world coordinates; WorldRoot IS the world origin in Unity, so apply
        // it as a local pose under worldRoot (same convention as GripStateReceiver / ToolSpawner).
        if (worldRoot != null && arBox.transform.parent != worldRoot)
            arBox.transform.SetParent(worldRoot, false);
        arBox.transform.localPosition = latest.pos;
        // GripStateReceiver.Awake wraps the imported mesh children in their
        // visual correction, so this cloned root remains the logical frame.
        arBox.transform.localRotation = latest.rot;
        if (applyIncomingBoxSize)
            arBox.transform.localScale = latest.size;
        if (latest.color.HasValue)
            ApplyBoardColor(latest.color.Value);
        else
            ApplyProximityColor(latest.proximityState);
        SetRenderersVisible(_renderers, true);

        if (showTargetGripper && _targetGripperVisual != null)
        {
            if (worldRoot != null && _targetGripperVisual.transform.parent != worldRoot)
                _targetGripperVisual.transform.SetParent(worldRoot, false);

            // Python defines the board offset along Open3D/TCP-local +Z.
            // The project conversion maps Open3D (x,y,z) to Unity (x,z,y),
            // therefore that local offset axis is Unity-local +Y, not +Z.
            Vector3 gripperOffsetAxis = latest.rot * Vector3.up;
            _targetGripperVisual.transform.localPosition = latest.pos - targetGripperForwardOffset * gripperOffsetAxis;
            _targetGripperVisual.transform.localRotation = latest.rot;
            if (latest.gripperColor.HasValue)
                ApplyTargetGripperColor(latest.gripperColor.Value);
            SetRenderersVisible(_targetGripperRenderers, true);
        }

        if (!_loggedShow)
        {
            _loggedShow = true;
            Debug.Log($"[WorkholdingBoxReceiver] First box pose applied — localPos="
                      + $"{arBox.transform.localPosition}  scale={latest.size}  "
                      + $"renderers={( _renderers != null ? _renderers.Length : 0)}  "
                      + $"targetGripper={(_targetGripperVisual != null ? _targetGripperVisual.name : "<none>")}");
        }
    }

    private void ApplyBoardMaterial()
    {
        if (!overrideBoardMaterial || _renderers == null) return;

        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        if (shader == null) return;

        _boardMaterial = new Material(shader) { name = "WorkholdingHalfBoard_TransparentBlack" };
        Color color = boardTint;
        color.a = boardAlpha;
        _boardMaterial.color = color;

        if (_boardMaterial.HasProperty("_BaseColor")) _boardMaterial.SetColor("_BaseColor", color);
        if (_boardMaterial.HasProperty("_Color")) _boardMaterial.SetColor("_Color", color);
        if (_boardMaterial.HasProperty("_Surface")) _boardMaterial.SetFloat("_Surface", 1f);
        if (_boardMaterial.HasProperty("_Blend")) _boardMaterial.SetFloat("_Blend", 0f);
        if (_boardMaterial.HasProperty("_SrcBlend")) _boardMaterial.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
        if (_boardMaterial.HasProperty("_DstBlend")) _boardMaterial.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        if (_boardMaterial.HasProperty("_ZWrite")) _boardMaterial.SetFloat("_ZWrite", 0f);

        _boardMaterial.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        _boardMaterial.EnableKeyword("_ALPHABLEND_ON");
        _boardMaterial.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;

        foreach (var r in _renderers)
            if (r != null) r.sharedMaterial = _boardMaterial;
    }

    private void ApplyTargetGripperMaterial()
    {
        if (_targetGripperRenderers == null) return;

        if (overrideTargetGripperMaterial)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            if (shader == null) return;

            _targetGripperMaterial = new Material(shader) { name = "WorkholdingTargetGripper_Ghost" };
            if (_targetGripperMaterial.HasProperty("_Surface")) _targetGripperMaterial.SetFloat("_Surface", 1f);
            if (_targetGripperMaterial.HasProperty("_Blend")) _targetGripperMaterial.SetFloat("_Blend", 0f);
            if (_targetGripperMaterial.HasProperty("_SrcBlend")) _targetGripperMaterial.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            if (_targetGripperMaterial.HasProperty("_DstBlend")) _targetGripperMaterial.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            if (_targetGripperMaterial.HasProperty("_ZWrite")) _targetGripperMaterial.SetFloat("_ZWrite", 0f);

            _targetGripperMaterial.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            _targetGripperMaterial.EnableKeyword("_ALPHABLEND_ON");
            _targetGripperMaterial.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        }

        foreach (var r in _targetGripperRenderers)
        {
            if (r == null) continue;
            if (_targetGripperMaterial != null)
                r.sharedMaterial = _targetGripperMaterial;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
        }
        ApplyTargetGripperColor(targetGripperTint);
    }

    private void ApplyTargetGripperColor(Color color)
    {
        if (_targetGripperRenderers == null) return;
        color.a = targetGripperAlpha;

        if (_targetGripperMaterial != null)
        {
            _targetGripperMaterial.color = color;
            if (_targetGripperMaterial.HasProperty("_BaseColor"))
                _targetGripperMaterial.SetColor("_BaseColor", color);
            if (_targetGripperMaterial.HasProperty("_Color"))
                _targetGripperMaterial.SetColor("_Color", color);
        }

        if (_targetGripperColorBlock == null)
            _targetGripperColorBlock = new MaterialPropertyBlock();
        foreach (Renderer renderer in _targetGripperRenderers)
        {
            if (renderer == null) continue;
            renderer.GetPropertyBlock(_targetGripperColorBlock);
            _targetGripperColorBlock.SetColor("_BaseColor", color);
            _targetGripperColorBlock.SetColor("_Color", color);
            renderer.SetPropertyBlock(_targetGripperColorBlock);
        }
    }

    private void ApplyBoardColor(Color color)
    {
        if (_boardMaterial == null) return;
        color.a = boardAlpha;
        _boardMaterial.color = color;
        if (_boardMaterial.HasProperty("_BaseColor"))
            _boardMaterial.SetColor("_BaseColor", color);
        if (_boardMaterial.HasProperty("_Color"))
            _boardMaterial.SetColor("_Color", color);
        ApplyTargetGripperColor(color);
        _lastProximityState = "";
    }

    private void ApplyProximityColor(string state)
    {
        if (_boardMaterial == null || state == _lastProximityState) return;
        Color color = state == "reached" ? reachedColor
                    : state == "near" ? nearColor
                    : state == "far" ? farColor
                    : state == "black" ? blackColor
                    : boardTint;
        color.a = boardAlpha;
        _boardMaterial.color = color;
        if (_boardMaterial.HasProperty("_BaseColor"))
            _boardMaterial.SetColor("_BaseColor", color);
        if (_boardMaterial.HasProperty("_Color"))
            _boardMaterial.SetColor("_Color", color);
        ApplyTargetGripperColor(color);
        _lastProximityState = state;
        Debug.Log($"[WorkholdingBoxReceiver] Target color state -> {state}");
    }

    private static void SetRenderersVisible(Renderer[] renderers, bool visible)
    {
        if (renderers == null) return;
        foreach (var r in renderers)
            if (r != null) r.enabled = visible;
    }

    private void OnDestroy()
    {
        _running = false;
        _socket?.Close();
        if (_thread?.IsAlive == true) _thread.Join(500);
        if (_boardMaterial != null)
            Destroy(_boardMaterial);
        if (_targetGripperMaterial != null)
            Destroy(_targetGripperMaterial);
        if (_targetGripperVisual != null)
            Destroy(_targetGripperVisual);
        NetMQManager.UnregisterReceiver();
    }
}
