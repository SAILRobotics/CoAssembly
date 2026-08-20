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

    [Header("Parent")]
    public Transform worldRoot;

    [Header("NetMQ")]
    [SerializeField] private int port = 5012;

    [Serializable]
    private class GripMessage
    {
        public string   grip_state;
        public float[]  box_pos;
        public float[]  box_rot_xyzw;
        public float[]  box_size;
    }

    private class PoseData
    {
        public string    gripState;
        public Vector3   boxPos;
        public Quaternion boxRot;
        public Vector3   boxSize;
    }

    private SubscriberSocket  _socket;
    private Thread            _thread;
    private volatile bool     _running;
    private readonly ConcurrentQueue<PoseData> _queue = new();

    private string _prevGripState  = "";
    private bool   _prevIsGrabbed;
    private bool   _boxFrozen;

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
                while (_running)
                {
                    try
                    {
                        if (_socket.TryReceiveFrameString(
                                TimeSpan.FromMilliseconds(100), out string msg))
                        {
                            var d = JsonConvert.DeserializeObject<GripMessage>(msg);
                            if (d == null) continue;
                            _queue.Enqueue(new PoseData
                            {
                                gripState = d.grip_state,
                                boxPos    = new Vector3(d.box_pos[0], d.box_pos[1], d.box_pos[2]),
                                boxRot    = new Quaternion(d.box_rot_xyzw[0], d.box_rot_xyzw[1],
                                                           d.box_rot_xyzw[2], d.box_rot_xyzw[3]),
                                boxSize   = new Vector3(d.box_size[0], d.box_size[1], d.box_size[2]),
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
        PoseData latest = null;
        while (_queue.TryDequeue(out var d)) latest = d;
        if (latest == null) return;

        bool handleActive = latest.gripState == "grabbed" || latest.gripState == "moving";
        bool grabbed      = manipulationHandle != null && manipulationHandle.IsGrabbed;
        bool moveComplete = latest.gripState == "grabbed" && _prevGripState == "moving";
        bool cancelled    = latest.gripState == "idle";

        // Freeze the box the instant the user releases — before Python even transitions to 'moving'
        if (_prevIsGrabbed && !grabbed) _boxFrozen = true;
        if (moveComplete || cancelled)  _boxFrozen = false;

        // Grip mode cancelled — hide everything and reset
        if (cancelled)
        {
            if (arBox    != null) arBox.SetActive(false);
            if (arHandle != null) arHandle.SetActive(false);
            _prevGripState = "";
            _prevIsGrabbed = false;
            return;
        }

        // Update box transform to follow claw only while idle and not frozen at a target
        if (arBox != null && !grabbed && !_boxFrozen)
        {
            arBox.transform.SetParent(worldRoot, false);
            arBox.transform.localPosition = latest.boxPos;
            arBox.transform.localRotation = latest.boxRot;
            if (applyIncomingBoxSize)
                arBox.transform.localScale = latest.boxSize;
        }

        // Hide box when robot finishes moving; ARManipulationHandle shows it on each grab
        if (moveComplete && arBox != null)
            arBox.SetActive(false);

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
                Vector3 boardNormal = arBox.transform.rotation * Vector3.right;
                float halfDepth = manipulationHandle != null
                    ? manipulationHandle.BoardHalfThickness : 0.0075f;
                float handleHalfDepth = manipulationHandle != null ? manipulationHandle.HandleHalfDepth : 0f;
                arHandle.transform.position = arBox.transform.position
                    - boardNormal * (halfDepth + handleHalfDepth);
                arHandle.transform.rotation = arBox.transform.rotation
                    * Quaternion.AngleAxis(-90f, Vector3.up);
            }
        }

        _prevGripState  = latest.gripState;
        _prevIsGrabbed  = grabbed;
    }

    private void OnDestroy()
    {
        _running = false;
        _socket?.Close();
        if (_thread?.IsAlive == true) _thread.Join(500);
        NetMQManager.UnregisterReceiver();
    }
}
