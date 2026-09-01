using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Receives grip state + box pose from Python (port 5012).
/// Positions and shows/hides the AR box and handle accordingly.
///
/// Inspector setup:
///   arBox       — the generic AR box GameObject (child of WorldRoot)
///   arHandle    — the AR manipulation handle GameObject (child of WorldRoot)
///   worldRoot   — WorldRoot transform (same as ToolSpawner)
/// </summary>
public class GripStateReceiver : MonoBehaviour
{
    [Header("AR objects")]
    public GameObject          arBox;
    public bool                applyIncomingBoxSize = true;
    public GameObject          arHandle;
    public ARManipulationHandle manipulationHandle;

    [Header("Carried-board gripper")]
    [Tooltip("Template cloned for a gripper that follows the participant-moved AR board.")]
    public GameObject          carriedGripperTemplate;
    public bool                showCarriedGripper = true;
    [Tooltip("Board-center offset from the TCP/gripper along Unity-local +Y.")]
    public float               carriedGripperForwardOffset = 0.23215f;

    [Header("Parent")]
    public Transform worldRoot;

    [Header("Target reached")]
    [Tooltip("Disable when this receiver represents the carried board and a separate target receiver owns status colors.")]
    public bool colorizeBoardState = true;
    [Min(0f)] public float targetPositionTolerance = 0.05f;
    [Range(0f, 180f)] public float targetAngleToleranceDeg = 15f;
    [Min(0f)] public float resultColorDuration = 0.75f;
    public Color manipulatingColor = Color.cyan;
    public Color movingColor = Color.yellow;
    public Color reachedColor = Color.green;
    public Color failedColor = Color.red;

    [Header("Target overlay")]
    [Tooltip("Use a transparent, non-depth-writing material so passthrough remains visible through the released target board.")]
    public bool transparentTargetOverlay = true;
    [Range(0f, 1f)] public float targetOverlayAlpha = 0.28f;

    [Header("NetMQ")]
    [SerializeField] private int port = 5012;

    [Serializable]
    private class GripMessage
    {
        public string   grip_state;
        public float[]  box_pos;
        public float[]  box_rot_xyzw;
        public float[]  box_size;
        public float[]  box_color;
    }

    private class PoseData
    {
        public string    gripState;
        public Vector3   boxPos;
        public Quaternion boxRot;
        public Vector3   boxSize;
        public Color?    boxColor;
    }

    private SubscriberSocket  _socket;
    private Thread            _thread;
    private volatile bool     _running;
    private readonly ConcurrentQueue<PoseData> _queue = new();

    private string _prevGripState = "";
    private bool   _prevIsGrabbed;
    private bool   _boxFrozen;
    private bool   _resultVisible;
    private bool   _targetReachedForCurrentMove;
    private float  _hideBoxAt;
    private Renderer[] _boardRenderers;
    private MaterialPropertyBlock _boardColorBlock;
    private Material _targetOverlayMaterial;
    private GameObject _carriedGripper;
    private Renderer[] _carriedGripperRenderers;
    private bool _loggedCarriedGripperVisible;

    private static readonly Quaternion BoardMeshCorrection =
        Quaternion.AngleAxis(90f, Vector3.forward)
        * Quaternion.AngleAxis(90f, Vector3.right);

    private void Awake()
    {
        if (arBox == null || arBox.transform.Find("HalfBoardVisualCorrection") != null)
            return;

        // Keep arBox itself as the logical board frame used by the receiver,
        // manipulation code, and pose publisher. Rotate only the imported OBJ
        // hierarchy so its dimensions/orientation match the Open3D rendering.
        Transform boardRoot = arBox.transform;
        int childCount = boardRoot.childCount;
        Transform[] importedChildren = new Transform[childCount];
        for (int i = 0; i < childCount; ++i)
            importedChildren[i] = boardRoot.GetChild(i);

        GameObject correctionObject = new GameObject("HalfBoardVisualCorrection");
        Transform correction = correctionObject.transform;
        correction.SetParent(boardRoot, false);
        correction.localRotation = BoardMeshCorrection;
        foreach (Transform child in importedChildren)
            child.SetParent(correction, false);
    }

    private void Start()
    {
        if (arBox    != null) arBox.SetActive(false);
        if (arHandle != null) arHandle.SetActive(false);
        if (carriedGripperTemplate != null)
        {
            Transform parent = worldRoot != null
                ? worldRoot : carriedGripperTemplate.transform.parent;
            _carriedGripper = Instantiate(
                carriedGripperTemplate, parent, false);
            _carriedGripper.name = "CarriedBoardGripper";
            _carriedGripperRenderers =
                _carriedGripper.GetComponentsInChildren<Renderer>(true);
            foreach (MonoBehaviour behaviour in
                     _carriedGripper.GetComponentsInChildren<MonoBehaviour>(true))
                behaviour.enabled = false;
            foreach (Collider collider in
                     _carriedGripper.GetComponentsInChildren<Collider>(true))
                collider.enabled = false;
            foreach (Rigidbody body in
                     _carriedGripper.GetComponentsInChildren<Rigidbody>(true))
            {
                body.isKinematic = true;
                body.detectCollisions = false;
            }
            _carriedGripper.SetActive(false);
            SetRenderersVisible(_carriedGripperRenderers, false);
            Debug.Log($"[GripStateReceiver] Created carried-board gripper "
                      + $"from {carriedGripperTemplate.name} with "
                      + $"{_carriedGripperRenderers.Length} renderer(s)");
        }
        if (arBox != null)
        {
            _boardRenderers = arBox.GetComponentsInChildren<Renderer>(true);
            _boardColorBlock = new MaterialPropertyBlock();
            ConfigureTargetOverlayMaterial();
        }

        NetMQManager.RegisterReceiver();
        _running = true;
        _thread  = new Thread(ReceiveLoop) { IsBackground = true };
        _thread.Start();
    }

    private void LateUpdate()
    {
        if (_carriedGripper == null) return;
        bool visible = showCarriedGripper && arBox != null
            && arBox.activeInHierarchy;
        _carriedGripper.SetActive(visible);
        SetRenderersVisible(_carriedGripperRenderers, visible);
        if (!visible) return;

        // Python's board offset is TCP-local +Z. The Open3D-to-Unity mapping
        // turns that into Unity-local +Y, matching WorkholdingBoxReceiver.
        Vector3 offsetAxis = arBox.transform.rotation * Vector3.up;
        _carriedGripper.transform.position = arBox.transform.position
            - carriedGripperForwardOffset * offsetAxis;
        _carriedGripper.transform.rotation = arBox.transform.rotation;
        if (!_loggedCarriedGripperVisible)
        {
            _loggedCarriedGripperVisible = true;
            Debug.Log($"[GripStateReceiver] Carried-board gripper visible at "
                      + $"{_carriedGripper.transform.position}");
        }
    }

    private static void SetRenderersVisible(Renderer[] renderers, bool visible)
    {
        if (renderers == null) return;
        foreach (Renderer renderer in renderers)
            if (renderer != null) renderer.enabled = visible;
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
                while (_running)
                {
                    try
                    {
                        if (_socket.TryReceiveFrameString(
                                TimeSpan.FromMilliseconds(100), out string msg))
                        {
                            var d = JsonConvert.DeserializeObject<GripMessage>(msg);
                            if (d == null) continue;
                            Color? incomingColor = null;
                            if (d.box_color != null && d.box_color.Length >= 3)
                            {
                                float alpha = d.box_color.Length >= 4
                                    ? d.box_color[3] : 1f;
                                incomingColor = new Color(
                                    d.box_color[0], d.box_color[1],
                                    d.box_color[2], alpha);
                            }
                            _queue.Enqueue(new PoseData
                            {
                                gripState = d.grip_state,
                                boxPos    = new Vector3(d.box_pos[0], d.box_pos[1], d.box_pos[2]),
                                boxRot    = new Quaternion(d.box_rot_xyzw[0], d.box_rot_xyzw[1],
                                                           d.box_rot_xyzw[2], d.box_rot_xyzw[3]),
                                boxSize   = new Vector3(d.box_size[0], d.box_size[1], d.box_size[2]),
                                boxColor  = incomingColor,
                            });
                        }
                    }
                    catch (TerminatingException)    { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception e) { if (_running) Debug.LogWarning("[GripStateReceiver] " + e.Message); }
                }
            }
        }
        catch (Exception e) { if (_running) Debug.LogWarning("[GripStateReceiver] Outer: " + e.Message); }
    }

    private void Update()
    {
        if (_resultVisible && Time.time >= _hideBoxAt)
        {
            _resultVisible = false;
            if (arBox != null) arBox.SetActive(false);
        }

        PoseData latest = null;
        while (_queue.TryDequeue(out var d)) latest = d;
        if (latest == null) return;

        bool handleActive = latest.gripState == "grabbed" || latest.gripState == "moving";
        bool grabbed      = manipulationHandle != null && manipulationHandle.IsGrabbed;
        bool moveComplete = latest.gripState == "grabbed" && _prevGripState == "moving";
        // A queued idle packet must never hide the board after ISDK has
        // already begun a grab. The local grab state is authoritative until
        // the manipulation ends.
        bool cancelled    = latest.gripState == "idle" && !grabbed;

        if (grabbed && arBox != null)
        {
            arBox.SetActive(true);
            SetRenderersVisible(_boardRenderers, true);
        }

        // Freeze the box the instant the user releases — before Python even transitions to 'moving'
        if (grabbed && !_prevIsGrabbed)
        {
            _resultVisible = false;
            _targetReachedForCurrentMove = false;
            SetBoardColor(manipulatingColor);
        }
        if (_prevIsGrabbed && !grabbed)
        {
            _boxFrozen = true;
            _resultVisible = false;
            _targetReachedForCurrentMove = false;
            if (arBox != null) arBox.SetActive(true);
            SetBoardColor(movingColor);
        }
        if (cancelled) _boxFrozen = false;

        // Grip mode cancelled — hide everything and reset
        if (cancelled)
        {
            if (manipulationHandle != null)
                manipulationHandle.CancelManipulation();
            if (arBox    != null) arBox.SetActive(false);
            if (arHandle != null) arHandle.SetActive(false);
            _prevGripState = "";
            _prevIsGrabbed = false;
            _resultVisible = false;
            _targetReachedForCurrentMove = false;
            return;
        }

        // Keep the released board frozen as the desired target while the
        // robot moves. It is hidden below when reached or when motion ends.
        bool targetReached = false;
        if (_boxFrozen && arBox != null)
        {
            float positionError = Vector3.Distance(
                arBox.transform.localPosition, latest.boxPos);
            float angleError = Quaternion.Angle(
                arBox.transform.localRotation, latest.boxRot);
            targetReached = positionError < targetPositionTolerance
                && angleError < targetAngleToleranceDeg;
            if (targetReached)
            {
                _boxFrozen = false;
                _targetReachedForCurrentMove = true;
                ShowResultColor(reachedColor);
                Debug.Log($"[GripStateReceiver] Target reached — "
                    + $"{positionError * 100f:F1} cm / {angleError:F1} deg");
            }
        }

        // moving → grabbed means the server ended the motion. If the pose did
        // not satisfy the target criterion, treat it as a failed move and
        // remove the stale target visualization as well.
        if (moveComplete && !_targetReachedForCurrentMove && arBox != null)
        {
            _boxFrozen = false;
            ShowResultColor(failedColor);
            Debug.Log("[GripStateReceiver] Robot move ended before target "
                + "was reached — hiding target");
        }
        if (moveComplete)
            _targetReachedForCurrentMove = false;

        // While AR control is available but no manipulation/move result is
        // active, expose only the selectable handle. The board and cloned
        // carried gripper appear when OnGrabbed activates arBox. In the
        // Hybrid freedrive-only zone, idle hides the handle as well.
        if (arBox != null && handleActive && !grabbed
                && !_boxFrozen && !_resultVisible)
            arBox.SetActive(false);

        // Update box transform to follow claw only while idle and not frozen at a target
        if (arBox != null && !grabbed && !_boxFrozen && !_resultVisible)
        {
            arBox.transform.SetParent(worldRoot, false);
            arBox.transform.localPosition = latest.boxPos;
            arBox.transform.localRotation = latest.boxRot;
            if (applyIncomingBoxSize)
                arBox.transform.localScale = latest.boxSize;
        }

        // Keep the handle on the box's near face whenever the box follows the
        // live TCP (including physical freedrive). While the user is grabbing
        // it, or while a released BoardAR target is frozen for robot motion,
        // ARManipulationHandle owns the pose instead.
        if (arHandle != null)
        {
            arHandle.SetActive(handleActive);
            if (handleActive && !grabbed && !_boxFrozen && arBox != null)
            {
                arHandle.transform.SetParent(worldRoot, false);
                arHandle.transform.position = arBox.transform.position
                    - arBox.transform.rotation * Vector3.right * 0.0075f
                    - arBox.transform.rotation * Vector3.forward * 0.2000f;
                arHandle.transform.rotation = arBox.transform.rotation
                    * Quaternion.AngleAxis(180f, Vector3.right);
            }
        }

        // An explicitly supplied Python RGBA value is authoritative and is
        // applied last so it overrides the receiver's optional local colors.
        if (latest.boxColor.HasValue)
            ApplyPythonBoardColor(latest.boxColor.Value);

        _prevGripState = latest.gripState;
        _prevIsGrabbed  = grabbed;
    }

    private void ShowResultColor(Color color)
    {
        if (arBox == null) return;
        arBox.SetActive(true);
        SetBoardColor(color);
        _resultVisible = true;
        _hideBoxAt = Time.time + resultColorDuration;
    }

    private void SetBoardColor(Color color)
    {
        if (!colorizeBoardState) return;
        if (_boardRenderers == null || _boardColorBlock == null) return;
        foreach (Renderer renderer in _boardRenderers)
        {
            if (renderer == null) continue;
            renderer.GetPropertyBlock(_boardColorBlock);
            _boardColorBlock.SetColor("_BaseColor", color);
            _boardColorBlock.SetColor("_Color", color);
            renderer.SetPropertyBlock(_boardColorBlock);
        }
    }

    private void ConfigureTargetOverlayMaterial()
    {
        if (!transparentTargetOverlay || _boardRenderers == null) return;
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        if (shader == null) return;

        _targetOverlayMaterial = new Material(shader)
            { name = "BoardARTarget_TransparentOverlay" };
        if (_targetOverlayMaterial.HasProperty("_Surface"))
            _targetOverlayMaterial.SetFloat("_Surface", 1f);
        if (_targetOverlayMaterial.HasProperty("_Blend"))
            _targetOverlayMaterial.SetFloat("_Blend", 0f);
        if (_targetOverlayMaterial.HasProperty("_SrcBlend"))
            _targetOverlayMaterial.SetFloat("_SrcBlend",
                (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
        if (_targetOverlayMaterial.HasProperty("_DstBlend"))
            _targetOverlayMaterial.SetFloat("_DstBlend",
                (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        if (_targetOverlayMaterial.HasProperty("_ZWrite"))
            _targetOverlayMaterial.SetFloat("_ZWrite", 0f);
        _targetOverlayMaterial.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        _targetOverlayMaterial.EnableKeyword("_ALPHABLEND_ON");
        _targetOverlayMaterial.renderQueue =
            (int)UnityEngine.Rendering.RenderQueue.Transparent;

        foreach (Renderer renderer in _boardRenderers)
        {
            if (renderer == null) continue;
            renderer.sharedMaterial = _targetOverlayMaterial;
            renderer.shadowCastingMode =
                UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.receiveShadows = false;
        }
    }

    private void ApplyPythonBoardColor(Color color)
    {
        if (_boardRenderers == null || _boardColorBlock == null) return;
        if (transparentTargetOverlay) color.a = targetOverlayAlpha;
        foreach (Renderer renderer in _boardRenderers)
        {
            if (renderer == null) continue;
            renderer.GetPropertyBlock(_boardColorBlock);
            _boardColorBlock.SetColor("_BaseColor", color);
            _boardColorBlock.SetColor("_Color", color);
            renderer.SetPropertyBlock(_boardColorBlock);
        }
    }

    private void OnDestroy()
    {
        _running = false;
        _socket?.Close();
        if (_thread?.IsAlive == true) _thread.Join(500);
        if (_targetOverlayMaterial != null) Destroy(_targetOverlayMaterial);
        NetMQManager.UnregisterReceiver();
    }
}
