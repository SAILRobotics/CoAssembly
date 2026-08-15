using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;

/// <summary>
/// Publishes live gearbox part poses to Python/Open3D.
/// Attach to the Gearbox Assembly Named Scripts root. Assign gearboxRoot to the
/// gearbox part hierarchy and worldRoot to the scene WorldRoot.
/// </summary>
public class GearboxPosePublisher : MonoBehaviour
{
    [SerializeField] private int port = 5027;
    [SerializeField] private Transform gearboxRoot;
    [SerializeField] private Transform worldRoot;
    [SerializeField] private float publishHz = 20f;
    [SerializeField] private bool includeInactive = true;

    [Serializable]
    private class PartPose
    {
        public string name;
        public bool active;
        public float[] pos;
        public float[] rot_xyzw;
        public float[] scale;
    }

    [Serializable]
    private class GearboxPoseMessage
    {
        public string type = "gearbox_pose";
        public PartPose[] parts;
    }

    private class PartEntry
    {
        public string name;
        public Transform transform;
        public Renderer[] renderers;
        public bool exactName;
    }

    private readonly List<PartEntry> parts = new();
    private PublisherSocket socket;
    private bool senderRegistered;
    private bool firstPublishLogged;
    private float nextSendAt;

    private void Awake()
    {
        Debug.Log($"[GearboxPosePublisher] Awake on {name}, enabled={enabled}, activeInHierarchy={gameObject.activeInHierarchy}");
    }

    private void OnEnable()
    {
        Debug.Log($"[GearboxPosePublisher] OnEnable on {name}");
    }

    private void Start()
    {
        if (gearboxRoot == null) gearboxRoot = transform;

        try
        {
            AsyncIO.ForceDotNet.Force();
            socket = new PublisherSocket();
            socket.Bind($"tcp://0.0.0.0:{port}");
            NetMQManager.RegisterSender();
            senderRegistered = true;
            Debug.Log($"[GearboxPosePublisher] PUB bound on tcp://0.0.0.0:{port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[GearboxPosePublisher] failed to bind tcp://0.0.0.0:{port}: {e}");
            try { socket?.Close(); socket?.Dispose(); } catch { }
            socket = null;
            return;
        }

        try
        {
            BuildIndex();
            Debug.Log($"[GearboxPosePublisher] indexed {parts.Count} logical parts under '{gearboxRoot.name}' worldRoot='{(worldRoot ? worldRoot.name : "none")}' includeInactive={includeInactive}");
        }
        catch (Exception e)
        {
            parts.Clear();
            Debug.LogError($"[GearboxPosePublisher] failed to index gearbox parts under '{gearboxRoot.name}': {e}");
        }
    }

    private void BuildIndex()
    {
        parts.Clear();
        var byName = new Dictionary<string, PartEntry>();

        foreach (var t in gearboxRoot.GetComponentsInChildren<Transform>(includeInactive))
        {
            string canonical = CanonicalPartName(t.name);
            if (!IsPartName(canonical)) continue;

            bool exact = t.name == canonical;
            if (byName.TryGetValue(canonical, out PartEntry existing) && existing.exactName && !exact)
            {
                continue;
            }

            byName[canonical] = new PartEntry
            {
                name = canonical,
                transform = t,
                renderers = t.GetComponentsInChildren<Renderer>(includeInactive),
                exactName = exact,
            };
        }

        parts.AddRange(byName.Values);
        parts.Sort((a, b) => string.CompareOrdinal(a.name, b.name));
    }

    private static string CanonicalPartName(string rawName)
    {
        const string occurrencePrefix = "occurrence of ";
        string n = rawName.Trim();
        while (n.StartsWith(occurrencePrefix, StringComparison.OrdinalIgnoreCase))
        {
            n = n.Substring(occurrencePrefix.Length).Trim();
        }
        return n;
    }

    private static bool IsPartName(string n)
    {
        string clean = n.Replace("_", "");
        return clean.Contains("Row") || n.IndexOf("BaseBoard", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private void LateUpdate()
    {
        if (socket == null || Time.time < nextSendAt) return;
        nextSendAt = Time.time + 1f / Mathf.Max(1f, publishHz);
        Publish();
    }

    private void Publish()
    {
        var poses = new PartPose[parts.Count];
        for (int i = 0; i < parts.Count; i++)
        {
            var part = parts[i];
            Transform t = part.transform;
            PoseLocalToWorldRoot(t, out Vector3 pos, out Quaternion rot);
            poses[i] = new PartPose
            {
                name = part.name,
                active = IsActive(part),
                pos = new[] { pos.x, pos.y, pos.z },
                rot_xyzw = new[] { rot.x, rot.y, rot.z, rot.w },
                scale = new[] { 1f, 1f, 1f },
            };
        }

        var msg = new GearboxPoseMessage { parts = poses };
        try
        {
            socket.SendFrame(JsonConvert.SerializeObject(msg));
            if (!firstPublishLogged)
            {
                Debug.Log($"[GearboxPosePublisher] first pose frame sent with {poses.Length} logical parts");
                firstPublishLogged = true;
            }
        }
        catch (Exception e) { Debug.LogWarning("[GearboxPosePublisher] " + e.Message); }
    }

    private static bool IsActive(PartEntry part)
    {
        if (!part.transform.gameObject.activeInHierarchy) return false;
        if (part.renderers == null || part.renderers.Length == 0) return true;
        foreach (var r in part.renderers)
        {
            if (r != null && r.enabled && r.gameObject.activeInHierarchy) return true;
        }
        return false;
    }

    private void PoseLocalToWorldRoot(Transform t, out Vector3 pos, out Quaternion rot)
    {
        if (worldRoot == null)
        {
            pos = t.position;
            rot = t.rotation;
            return;
        }

        pos = worldRoot.InverseTransformPoint(t.position);
        rot = Quaternion.Inverse(worldRoot.rotation) * t.rotation;
    }

    private void OnDestroy()
    {
        if (socket != null)
        {
            try { socket.Close(); socket.Dispose(); } catch { }
            socket = null;
        }
        if (senderRegistered)
        {
            NetMQManager.UnregisterSender();
            senderRegistered = false;
        }
    }
}