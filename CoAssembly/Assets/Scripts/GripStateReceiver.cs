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

    private string _prevGripState = "";

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

        bool handleActive = latest.gripState == "grabbed" || latest.gripState == "moving_to_pose";

        // Position the box only when not grabbed — ARManipulationHandle owns it while grabbed
        bool grabbed = manipulationHandle != null && manipulationHandle.IsGrabbed;
        if (arBox != null && !grabbed)
        {
            arBox.transform.SetParent(worldRoot, false);
            arBox.transform.localPosition = latest.boxPos;
            arBox.transform.localRotation = latest.boxRot;
            arBox.transform.localScale    = latest.boxSize;
        }

        // Position the handle on the near face of the box only on first arrival ('grabbed' after null/unknown).
        // After a manipulation move ('moving_to_pose' → 'grabbed'), leave it at the released position.
        if (arHandle != null)
        {
            arHandle.SetActive(handleActive);
            bool firstArrival = latest.gripState == "grabbed"
                                && _prevGripState != "grabbed"
                                && _prevGripState != "moving_to_pose";
            if (firstArrival && !grabbed && arBox != null)
            {
                arHandle.transform.SetParent(worldRoot, false);
                Vector3 boxForward = arBox.transform.rotation * Vector3.forward;
                float   halfDepth  = latest.boxSize.z * 0.5f;
                arHandle.transform.position = arBox.transform.position - boxForward * halfDepth;
                arHandle.transform.rotation = arBox.transform.rotation;
            }
        }

        _prevGripState = latest.gripState;
    }

    private void OnDestroy()
    {
        _running = false;
        _socket?.Close();
        if (_thread?.IsAlive == true) _thread.Join(500);
        NetMQManager.UnregisterReceiver();
    }
}
