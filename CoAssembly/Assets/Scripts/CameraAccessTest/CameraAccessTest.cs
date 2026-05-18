using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using Meta.XR;

public class PassthroughCameraPublisher : MonoBehaviour
{
    [Header("Assign in Inspector")]
    public PassthroughCameraAccess leftCamera;
    public PassthroughCameraAccess rightCamera;

    [Header("NetMQ")]
    public int port = 5560;
    public float sendHz = 15f;
    [Range(1, 100)] public int jpegQuality = 60;

    [Header("Send flags")]
    public bool sendLeft = true;
    public bool sendRight = false;

    [Header("Debug")]
    public bool verboseLogs = true;

    private PublisherSocket pub;
    private Thread sendThread;
    private volatile bool running;

    private readonly ConcurrentQueue<FramePacket> queue = new();
    private float nextSendTime = 0f;
    private bool registeredSender = false;

    private struct FramePacket
    {
        public string topic;
        public long timestampMs;
        public int width;
        public int height;

        public float px;
        public float py;
        public float pz;

        public float qx;
        public float qy;
        public float qz;
        public float qw;

        public float fx;
        public float fy;
        public float cx;
        public float cy;

        public int sensorWidth;
        public int sensorHeight;

        public byte[] jpeg;
    }

    void Start()
    {
        AsyncIO.ForceDotNet.Force();

        NetMQManager.RegisterSender();
        registeredSender = true;

        running = true;
        sendThread = new Thread(SendLoop)
        {
            IsBackground = true,
            Name = "PassthroughCameraPublisher_SendLoop"
        };
        sendThread.Start();

        if (verboseLogs)
            Debug.Log($"[PUB] Started publisher on tcp://0.0.0.0:{port}");
    }

    void Update()
    {
        if (!running) return;
        if (NetMQManager.IsShutdownRequested) return;
        if (sendHz <= 0f) return;
        if (Time.unscaledTime < nextSendTime) return;

        nextSendTime = Time.unscaledTime + 1f / sendHz;

        if (sendLeft && leftCamera != null)
            TryQueueCamera(leftCamera, "cam_left");

        if (sendRight && rightCamera != null)
            TryQueueCamera(rightCamera, "cam_right");
    }

    private void TryQueueCamera(PassthroughCameraAccess cam, string topic)
    {
        if (!running) return;
        if (NetMQManager.IsShutdownRequested) return;

        if (cam == null || !cam.enabled)
            return;

        if (!cam.IsPlaying)
        {
            if (verboseLogs)
                Debug.Log($"[PUB] {topic}: camera not playing yet");
            return;
        }

        if (!cam.IsUpdatedThisFrame)
        {
            if (verboseLogs)
                Debug.Log($"[PUB] {topic}: no fresh frame this Unity frame");
            return;
        }

        var src = cam.GetTexture() as RenderTexture;
        if (src == null)
        {
            if (verboseLogs)
                Debug.LogWarning($"[PUB] {topic}: GetTexture() is null or not a RenderTexture");
            return;
        }

        int w = src.width;
        int h = src.height;

        Pose camPose;
        PassthroughCameraAccess.CameraIntrinsics intr;
        try
        {
            camPose = cam.GetCameraPose();
            intr = cam.Intrinsics;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PUB] {topic}: failed to get camera pose/intrinsics: {e}");
            return;
        }

        Vector3 p = camPose.position;
        Quaternion q = camPose.rotation;

        Vector2 focal = intr.FocalLength;
        Vector2 principal = intr.PrincipalPoint;
        Vector2Int sensorRes = intr.SensorResolution;

        AsyncGPUReadback.Request(src, 0, request =>
        {
            if (!running || NetMQManager.IsShutdownRequested)
                return;

            if (request.hasError)
            {
                Debug.LogWarning($"[PUB] {topic}: AsyncGPUReadback error");
                return;
            }

            try
            {
                NativeArray<Color32> raw = request.GetData<Color32>();

                Texture2D tex = new Texture2D(w, h, TextureFormat.RGBA32, false, false);
                tex.LoadRawTextureData(raw);
                tex.Apply(false, false);

                byte[] jpg = tex.EncodeToJPG(jpegQuality);
                Destroy(tex);

                long tsMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

                queue.Enqueue(new FramePacket
                {
                    topic = topic,
                    timestampMs = tsMs,
                    width = w,
                    height = h,

                    px = p.x,
                    py = p.y,
                    pz = p.z,

                    qx = q.x,
                    qy = q.y,
                    qz = q.z,
                    qw = q.w,

                    fx = focal.x,
                    fy = focal.y,
                    cx = principal.x,
                    cy = principal.y,

                    sensorWidth = sensorRes.x,
                    sensorHeight = sensorRes.y,

                    jpeg = jpg
                });

                if (verboseLogs)
                {
                    Debug.Log(
                        $"[PUB] {topic}: queued jpg {jpg.Length} bytes with pose + intrinsics " +
                        $"fx={focal.x:F2}, fy={focal.y:F2}, cx={principal.x:F2}, cy={principal.y:F2}, " +
                        $"sensor=({sensorRes.x},{sensorRes.y})"
                    );
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[PUB] {topic}: exception during readback/encode: {e}");
            }
        });
    }

    private void SendLoop()
    {
        try
        {
            using (pub = new PublisherSocket())
            {
                pub.Options.SendHighWatermark = 1;
                pub.Bind($"tcp://0.0.0.0:{port}");

                while (running && !NetMQManager.IsShutdownRequested)
                {
                    while (queue.TryDequeue(out FramePacket pkt))
                    {
                        byte[] tsBytes = BitConverter.GetBytes(pkt.timestampMs);
                        byte[] wBytes = BitConverter.GetBytes(pkt.width);
                        byte[] hBytes = BitConverter.GetBytes(pkt.height);

                        byte[] posBytes = new byte[12];
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.px), 0, posBytes, 0, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.py), 0, posBytes, 4, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.pz), 0, posBytes, 8, 4);

                        byte[] rotBytes = new byte[16];
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.qx), 0, rotBytes, 0, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.qy), 0, rotBytes, 4, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.qz), 0, rotBytes, 8, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.qw), 0, rotBytes, 12, 4);

                        byte[] intrBytes = new byte[16];
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.fx), 0, intrBytes, 0, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.fy), 0, intrBytes, 4, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.cx), 0, intrBytes, 8, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.cy), 0, intrBytes, 12, 4);

                        byte[] sensorBytes = new byte[8];
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.sensorWidth), 0, sensorBytes, 0, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(pkt.sensorHeight), 0, sensorBytes, 4, 4);

                        pub.SendMoreFrame(pkt.topic)
                           .SendMoreFrame(tsBytes)
                           .SendMoreFrame(wBytes)
                           .SendMoreFrame(hBytes)
                           .SendMoreFrame(posBytes)
                           .SendMoreFrame(rotBytes)
                           .SendMoreFrame(intrBytes)
                           .SendMoreFrame(sensorBytes)
                           .SendFrame(pkt.jpeg);

                        if (verboseLogs)
                        {
                            Debug.Log($"[PUB] sent {pkt.topic} ts={pkt.timestampMs} size={pkt.jpeg.Length}B");
                        }
                    }

                    Thread.Sleep(1);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[PUB] SendLoop crashed: {e}");
        }
    }

    private void OnDestroy()
    {
        Shutdown();
    }

    private void OnApplicationQuit()
    {
        Shutdown();
    }

    private void Shutdown()
    {
        if (!running)
            return;

        running = false;

        try
        {
            sendThread?.Join(1000);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PUB] sendThread join exception: {e}");
        }

        try
        {
            pub?.Dispose();
        }
        catch { }

        pub = null;

        if (registeredSender)
        {
            NetMQManager.UnregisterSender();
            registeredSender = false;
        }

        if (verboseLogs)
            Debug.Log("[PUB] Shutdown complete");
    }
}