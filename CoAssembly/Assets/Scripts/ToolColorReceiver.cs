using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Newtonsoft.Json;

public class ToolColorReceiver : MonoBehaviour
{
    [Header("Tool Identity")]
    [SerializeField] private int toolId;

    [Header("Visual")]
    [SerializeField] private Renderer targetRenderer;

    [Header("NetMQ")]
    [SerializeField] private int port = 5010;

    [Serializable]
    private class ColorMessage
    {
        public int tool_id;
        public float[] color;
    }

    // ── Shared socket + dispatcher across all instances ──────────────────────
    private static SubscriberSocket sharedSocket;
    private static Thread sharedThread;
    private static volatile bool sharedRunning = false;
    private static int sharedRefCount = 0;
    private static readonly object sharedLock = new object();
    private static readonly Dictionary<int, ToolColorReceiver> instances = new();

    // ── Per-instance state ───────────────────────────────────────────────────
    private readonly ConcurrentQueue<Color> pendingColor = new();
    private MaterialPropertyBlock propertyBlock;
    private int colorID;
    private Color originalColor; 

    private void Start()
    {
        Debug.Log($"[ToolColorReceiver] 🟢 Start() called on GameObject '{gameObject.name}', toolId={toolId}");

        if (targetRenderer == null) targetRenderer = GetComponent<Renderer>();
        if (targetRenderer == null) targetRenderer = GetComponentInChildren<Renderer>();
        if (targetRenderer == null)
        {
            Debug.LogError($"[ToolColorReceiver] No Renderer on {name}");
            enabled = false;
            return;
        }

        Material mat = targetRenderer.sharedMaterial;
        colorID = mat.HasProperty("_BaseColor")
            ? Shader.PropertyToID("_BaseColor")
            : Shader.PropertyToID("_Color");
        propertyBlock = new MaterialPropertyBlock();
        originalColor = mat.GetColor(colorID);
        lock (sharedLock)
        {
            instances[toolId] = this;

            if (sharedSocket == null)
            {
                AsyncIO.ForceDotNet.Force();
                sharedSocket = new SubscriberSocket();
                sharedSocket.Bind($"tcp://0.0.0.0:{port}");
                sharedSocket.Subscribe("");
                NetMQManager.RegisterReceiver();

                sharedRunning = true;
                sharedThread = new Thread(SharedReceiveLoop) { IsBackground = true };
                sharedThread.Start();
                Debug.Log($"[ToolColorReceiver] 📡 Shared SUB bound on tcp://0.0.0.0:{port}");
            }
            sharedRefCount++;
        }
    }

    private static void SharedReceiveLoop()
    {
        try
        {
            while (sharedRunning)
            {
                try
                {
                    if (sharedSocket.TryReceiveFrameString(
                            TimeSpan.FromMilliseconds(100), out string message))
                    {
                        var data = JsonConvert.DeserializeObject<ColorMessage>(message);
                        if (data != null && data.color != null && data.color.Length >= 3)
                        {
                            Debug.Log($"[ToolColorReceiver] 📥 RX tool={data.tool_id}, " +
                                    $"length={data.color.Length}, " +
                                    $"raw=[{string.Join(",", data.color)}]"); 
                            ToolColorReceiver target;
                            lock (sharedLock)
                            {
                                instances.TryGetValue(data.tool_id, out target);
                            }
                            if (target == null) continue;

                            Color c;
                            if (data.color[0] < 0f)   // sentinel = restore original
                            {
                                c = target.originalColor;
                            }
                            else
                            {
                                float a = data.color.Length >= 4 ? data.color[3] : 1.0f;
                                c = new Color(data.color[0], data.color[1], data.color[2], a);
                            }
                            target.pendingColor.Enqueue(c);
                        }
                    }
                }
                catch (TerminatingException) { break; }
                catch (ObjectDisposedException) { break; }
                catch (Exception e)
                {
                    if (sharedRunning)
                        Debug.LogWarning($"[ToolColorReceiver shared] {e.Message}");
                }
            }
        }
        catch (Exception e)
        {
            if (sharedRunning)
                Debug.LogWarning($"[ToolColorReceiver shared] Outer: {e.Message}");
        }
    }

    private void Update()
    {
        while (pendingColor.TryDequeue(out Color c))
        {
            Debug.Log($"[ToolColorReceiver:{toolId}] 🎨 Applying ({c.r:F2},{c.g:F2},{c.b:F2},{c.a:F2}) " +
            $"to renderer '{targetRenderer.name}' on '{gameObject.name}'");
            targetRenderer.GetPropertyBlock(propertyBlock);
            propertyBlock.SetColor(colorID, c);
            targetRenderer.SetPropertyBlock(propertyBlock);
        }
    }

    private void OnDestroy()
    {
        lock (sharedLock)
        {
            instances.Remove(toolId);
            sharedRefCount = Mathf.Max(0, sharedRefCount - 1);

            if (sharedRefCount == 0 && sharedSocket != null)
            {
                sharedRunning = false;
                try { sharedSocket.Close(); sharedSocket.Dispose(); } catch { }
                sharedSocket = null;

                if (sharedThread != null && sharedThread.IsAlive)
                    sharedThread.Join(500);
                sharedThread = null;

                NetMQManager.UnregisterReceiver();
            }
        }
    }
}