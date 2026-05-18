using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Collections.Concurrent;
using System.Threading;

public class OVRControllerAndHeadPublisherNetMQ : MonoBehaviour
{
    [Header("Drag these from OVRCameraRig")]
    public Transform LeftControllerAnchor;
    public Transform RightControllerAnchor;
    public Transform CenterEyeAnchor;

    [Header("Optional eye anchors")]
    public Transform LeftEyeAnchor;
    public Transform RightEyeAnchor;

    [Header("NetMQ")]
    public int port = 5559;
    public float sendHz = 60f;
    public bool sendLeft = true;
    public bool sendRight = true;
    public bool sendHead = true;
    public bool sendLeftEye = true;
    public bool sendRightEye = true;

    private Thread sendThread;
    private volatile bool running;
    private bool hasShutdown = false;
    private PublisherSocket pub;
    private readonly ConcurrentQueue<string> queue = new();

    private float nextSendTime;

    [Serializable]
    private class PoseMsg
    {
        public bool is_valid;
        public float[] pos;
        public float[] rot_xyzw;
    }

    [Serializable]
    private class ControllerMsg : PoseMsg
    {
        public float trigger;
        public float grip;

        public bool trigger_btn;
        public bool grip_btn;

        public bool primary;
        public bool secondary;

        public float[] thumbstick;
        public bool thumbstick_click;

        public bool menu;
    }

    [Serializable]
    private class Msg
    {
        public double t_unity;

        public PoseMsg head;
        public PoseMsg left_eye;
        public PoseMsg right_eye;

        public ControllerMsg left;
        public ControllerMsg right;
    }

    void Start()
    {
        AsyncIO.ForceDotNet.Force();

        NetMQManager.RegisterSender();

        running = true;
        sendThread = new Thread(SendLoop)
        {
            IsBackground = true
        };
        sendThread.Start();

        Debug.Log($"[OVRPublisher] 📡 Publisher started on tcp://*:{port}");
    }

    void Update()
    {
        if (sendHz <= 0f) return;
        if (Time.unscaledTime < nextSendTime) return;
        nextSendTime = Time.unscaledTime + (1f / sendHz);

        var msg = new Msg
        {
            t_unity = Time.unscaledTimeAsDouble,

            head = sendHead ? SamplePose(CenterEyeAnchor) : null,
            left_eye = sendLeftEye ? SamplePose(LeftEyeAnchor) : null,
            right_eye = sendRightEye ? SamplePose(RightEyeAnchor) : null,

            left = sendLeft ? SampleController(isLeft: true) : null,
            right = sendRight ? SampleController(isLeft: false) : null
        };

        queue.Enqueue(JsonConvert.SerializeObject(msg));

        if (NetMQManager.IsShutdownRequested)
            Shutdown();
    }

    private PoseMsg SamplePose(Transform tf)
    {
        bool ok = tf != null;
        Vector3 p = ok ? tf.position : default;
        Quaternion q = ok ? tf.rotation : default;

        return new PoseMsg
        {
            is_valid = ok,
            pos = new[] { p.x, p.y, p.z },
            rot_xyzw = new[] { q.x, q.y, q.z, q.w }
        };
    }

    private ControllerMsg SampleController(bool isLeft)
    {
        Transform tf = isLeft ? LeftControllerAnchor : RightControllerAnchor;
        bool poseOk = tf != null;

        Vector3 pos = poseOk ? tf.position : default;
        Quaternion rot = poseOk ? tf.rotation : default;

        var ctrl = isLeft ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch;

        float trigger = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, ctrl);
        float grip = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, ctrl);

        bool triggerBtn = OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, ctrl);
        bool gripBtn = OVRInput.Get(OVRInput.Button.PrimaryHandTrigger, ctrl);

        bool primary = OVRInput.Get(OVRInput.Button.One, ctrl);
        bool secondary = OVRInput.Get(OVRInput.Button.Two, ctrl);

        Vector2 stick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, ctrl);
        bool stickClick = OVRInput.Get(OVRInput.Button.PrimaryThumbstick, ctrl);

        bool menu = isLeft && OVRInput.Get(OVRInput.Button.Start, ctrl);

        return new ControllerMsg
        {
            is_valid = poseOk,

            pos = new[] { pos.x, pos.y, pos.z },
            rot_xyzw = new[] { rot.x, rot.y, rot.z, rot.w },

            trigger = trigger,
            grip = grip,

            trigger_btn = triggerBtn,
            grip_btn = gripBtn,

            primary = primary,
            secondary = secondary,

            thumbstick = new[] { stick.x, stick.y },
            thumbstick_click = stickClick,

            menu = menu
        };
    }

    private void SendLoop()
    {
        try
        {
            using (pub = new PublisherSocket())
            {
                pub.Bind($"tcp://*:{port}");

                while (running)
                {
                    try
                    {
                        while (queue.TryDequeue(out var s))
                            pub.SendMoreFrame("xr").SendFrame(s);

                        Thread.Sleep(1);
                    }
                    catch (TerminatingException)
                    {
                        break;
                    }
                    catch (ObjectDisposedException)
                    {
                        break;
                    }
                    catch (Exception e)
                    {
                        if (running)
                            Debug.LogWarning("[OVRPublisher] ❌ SendLoop error: " + e.Message);
                    }
                }
            }
        }
        catch (Exception e)
        {
            if (running)
                Debug.LogWarning("[OVRPublisher] ❌ SendLoop outer exception: " + e.Message);
        }
    }

    private void Shutdown()
    {
        if (hasShutdown) return;
        hasShutdown = true;

        Debug.Log("[OVRPublisher] 🔻 Shutting down...");

        try
        {
            running = false;

            pub?.Close();
            pub?.Dispose();
            pub = null;

            if (sendThread != null && sendThread.IsAlive)
            {
                if (!sendThread.Join(1000))
                    Debug.LogWarning("[OVRPublisher] ⚠️ Send thread did not exit within timeout.");
            }

            NetMQManager.UnregisterSender();
            Debug.Log("[OVRPublisher] ✅ Shutdown complete");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[OVRPublisher] ⚠️ Shutdown exception: " + e.Message);
        }
    }

    private void OnDestroy() => Shutdown();
    private void OnApplicationQuit() => Shutdown();
}