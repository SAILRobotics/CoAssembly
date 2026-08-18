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
    public int toolId;

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
    private static readonly int BaseColorID = Shader.PropertyToID("_BaseColor");
    private static readonly int ColorID = Shader.PropertyToID("_Color");
    private static readonly int BaseColorFactorID = Shader.PropertyToID("_BaseColorFactor");
    private Color originalColor; 
    private bool hasExplicitTarget;
    private Renderer[] targetRenderers;
    private Renderer secondaryRenderer;
    private float secondaryAlpha = 0.06f;

    public void ConfigureVisual(int id, Renderer renderer, Renderer secondary = null, float secondaryAlpha = 0.06f)
    {
        ConfigureVisual(id, renderer != null ? new[] { renderer } : null, secondary, secondaryAlpha);
    }

    public void ConfigureVisual(int id, Renderer[] renderers, Renderer secondary = null, float secondaryAlpha = 0.06f)
    {
        toolId = id;
        targetRenderers = renderers;
        targetRenderer = renderers != null && renderers.Length > 0 ? renderers[0] : null;
        secondaryRenderer = secondary;
        this.secondaryAlpha = secondaryAlpha;
        hasExplicitTarget = targetRenderer != null;
    }

    private void Start()
    {
        Debug.Log($"[ToolColorReceiver] 🟢 Start() called on GameObject '{gameObject.name}', toolId={toolId}");

        hasExplicitTarget = targetRenderer != null;
        if (targetRenderer == null) targetRenderer = GetComponent<Renderer>();
        if (targetRenderer == null) targetRenderer = GetComponentInChildren<Renderer>();
        if (targetRenderer == null)
        {
            Debug.LogError($"[ToolColorReceiver] No Renderer on {name}");
            enabled = false;
            return;
        }

        if (targetRenderers == null || targetRenderers.Length == 0)
            targetRenderers = new[] { targetRenderer };

        Material mat = targetRenderer.sharedMaterial;
        propertyBlock = new MaterialPropertyBlock();
        if (mat.HasProperty(BaseColorID))
            originalColor = mat.GetColor(BaseColorID);
        else if (mat.HasProperty(ColorID))
            originalColor = mat.GetColor(ColorID);
        else if (mat.HasProperty(BaseColorFactorID))
            originalColor = mat.GetColor(BaseColorFactorID);
        else
            originalColor = Color.white;
        lock (sharedLock)
        {
            // Prefer an Inspector-assigned renderer (TCPMarker) over receivers
            // that merely auto-discovered a renderer. This makes duplicate IDs
            // deterministic while legacy scene components are being cleaned up.
            if (!instances.TryGetValue(toolId, out ToolColorReceiver existing)
                    || hasExplicitTarget || !existing.hasExplicitTarget)
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
            string rendererName = targetRenderer != null ? targetRenderer.name : "<none>";
            Debug.Log($"[ToolColorReceiver:{toolId}] Applying ({c.r:F2},{c.g:F2},{c.b:F2},{c.a:F2}) " +
            $"to renderer '{rendererName}' on '{gameObject.name}'");
            if (targetRenderers != null)
            {
                foreach (var renderer in targetRenderers)
                    ApplyColor(renderer, c);
            }
            if (secondaryRenderer != null)
            {
                Color secondaryColor = c;
                secondaryColor.a = Mathf.Min(c.a, secondaryAlpha);
                ApplyColor(secondaryRenderer, secondaryColor);
            }
        }
    }

    private void ApplyColor(Renderer renderer, Color color)
    {
        if (renderer == null) return;
        renderer.GetPropertyBlock(propertyBlock);
        propertyBlock.SetColor(BaseColorID, color);
        propertyBlock.SetColor(ColorID, color);
        propertyBlock.SetColor(BaseColorFactorID, color);
        renderer.SetPropertyBlock(propertyBlock);
    }

    private void OnDestroy()
    {
        lock (sharedLock)
        {
            if (instances.TryGetValue(toolId, out ToolColorReceiver registered)
                    && object.ReferenceEquals(registered, this))
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
