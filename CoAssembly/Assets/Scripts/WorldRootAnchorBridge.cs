using System;
using System.Collections.Generic;
using UnityEngine;
using Meta.XR.BuildingBlocks;

/// <summary>
/// Bridges the Python world-root lock signal to a Meta spatial anchor so the
/// scene stays fixed in physical space even when the Quest's tracking origin drifts.
///
/// Inspector setup
/// ---------------
///   worldRootReceiver  — WorldRootPoseReceiverNetMQ on the same or any GameObject
///   anchorCore         — SpatialAnchorCoreBuildingBlock added via Building Blocks panel
///   worldRoot          — the WorldRoot Transform (same one assigned in the receiver)
///
/// Lifecycle
/// ---------
///   First session : Python locks → OnPoseReceived fires → bridge creates anchor at that
///                   pose → WorldRoot is re-parented to the anchor at local-zero → UUID
///                   saved in PlayerPrefs.
///   Later sessions: bridge reads UUID on Start, loads anchor, re-parents WorldRoot
///                   without needing an ArUco lock.
///   Normal relocks: ignored once an anchor is active (the anchor already corrects drift).
///   ResetAnchor()  : call this to force a full re-anchor (e.g. after physically moving
///                   the ArUco marker). Destroys the current anchor and waits for the
///                   next Python lock signal.
/// </summary>
public class WorldRootAnchorBridge : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private WorldRootPoseReceiverNetMQ worldRootReceiver;
    [SerializeField] private SpatialAnchorCoreBuildingBlock anchorCore;
    [SerializeField] private Transform worldRoot;

    private const string UuidPrefKey = "WorldAnchorUUID";

    [Header("Initialization")]
    [Tooltip("Wait this long after app start before attempting first anchor creation (SLAM needs time to initialize).")]
    [SerializeField] private float initialDelayBeforeFirstAttempt = 3f;
    private float _initStartTime = -1f;

    [Header("Retry")]
    [Tooltip("Seconds between anchor creation retries after failure (SLAM may need time to load map).")]
    [SerializeField] private float retryDelaySecs = 4f;
    [Tooltip("Give up after this many failed attempts and fall back to direct positioning.")]
    [SerializeField] private int   maxRetries = 4;

    private Transform        _originalParent;
    private OVRSpatialAnchor _currentAnchor;
    private bool             _anchorActive;
    private bool             _creatingAnchor;
    private bool             _loadingAnchor;
    private float            _loadingStartTime;
    private bool             _anchorsUnavailable;
    private int              _retryCount;
    private float            _retryAfterTime = -1f;

    // Building block only fires OnAnchorsLoadCompleted on success — never on failure.
    // If loading hangs beyond this, assume failure and fall through to anchor creation.
    private const float LoadTimeoutSecs = 6f;

    // ── Unity lifecycle ───────────────────────────────────────────────────────

    private void Awake()
    {
        _originalParent = worldRoot != null ? worldRoot.parent : null;
    }

    private void Start()
    {
        if (worldRootReceiver == null || anchorCore == null || worldRoot == null)
        {
            Debug.LogError("[WorldRootAnchorBridge] Missing reference — check Inspector.");
            enabled = false;
            return;
        }

        Debug.Log("[WorldRootAnchorBridge] ========== STARTUP ==========");
        Debug.Log($"  initialDelayBeforeFirstAttempt={initialDelayBeforeFirstAttempt}s (SLAM initialization time)");
        Debug.Log($"  maxRetries={maxRetries}, retryDelaySecs={retryDelaySecs}s, LoadTimeoutSecs={LoadTimeoutSecs}s");
        Debug.Log($"  worldRootReceiver={worldRootReceiver.gameObject.name}");
        Debug.Log($"  anchorCore={anchorCore.gameObject.name}");
        Debug.Log($"  worldRoot={worldRoot.gameObject.name}");

        _initStartTime = Time.time;

        worldRootReceiver.OnPoseReceived += OnPoseReceived;
        anchorCore.OnAnchorCreateCompleted.AddListener(OnAnchorCreated);
        anchorCore.OnAnchorsLoadCompleted.AddListener(OnAnchorsLoaded);

        TryLoadSavedAnchor();
        Debug.Log("[WorldRootAnchorBridge] ========== Startup complete ==========");
    }

    private void Update()
    {
        if (_loadingAnchor && Time.time - _loadingStartTime > LoadTimeoutSecs)
        {
            _loadingAnchor = false;
            PlayerPrefs.DeleteKey(UuidPrefKey);
            PlayerPrefs.Save();
            Debug.LogWarning("[WorldRootAnchorBridge] Anchor load timed out (building block " +
                             "gave no failure callback) — clearing saved UUID, will create fresh anchor.");
        }
    }

    private void OnDestroy()
    {
        if (worldRootReceiver != null)
            worldRootReceiver.OnPoseReceived -= OnPoseReceived;

        if (anchorCore != null)
        {
            anchorCore.OnAnchorCreateCompleted.RemoveListener(OnAnchorCreated);
            anchorCore.OnAnchorsLoadCompleted.RemoveListener(OnAnchorsLoaded);
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Destroys the current anchor and clears the saved UUID so the next Python
    /// lock signal creates a fresh anchor. Call this if the ArUco marker has been
    /// physically moved or you want to fully re-anchor the scene.
    /// </summary>
    public void ResetAnchor()
    {
        if (_currentAnchor != null)
        {
            worldRoot.SetParent(_originalParent, worldPositionStays: true);
            anchorCore.EraseAllAnchors();   // safe here — we know an anchor exists
            Destroy(_currentAnchor.gameObject);
            _currentAnchor = null;
        }

        _anchorActive       = false;
        _creatingAnchor     = false;
        _anchorsUnavailable = false;
        _retryCount         = 0;
        _retryAfterTime     = -1f;

        PlayerPrefs.DeleteKey(UuidPrefKey);
        PlayerPrefs.Save();

        Debug.Log("[WorldRootAnchorBridge] Anchor reset — waiting for next ArUco lock.");
    }

    // ── Pose received from Python ─────────────────────────────────────────────

    private void OnPoseReceived(Vector3 pos, Quaternion rot)
    {
        // Position WorldRoot directly when anchors are unavailable on this device.
        if (_anchorsUnavailable)
        {
            worldRoot.SetPositionAndRotation(pos, rot);
            return;
        }

        // Once an anchor is active it corrects drift automatically — ignore
        // the continuous identical publishes Python sends every frame.
        // Also wait while a previous-session anchor is still localizing.
        if (_anchorActive || _loadingAnchor)
        {
            if (_anchorActive && _retryCount > 0)
            {
                Debug.Log($"[WorldRootAnchorBridge] Ignoring pose (anchor active, had {_retryCount} prior failed attempts)");
            }
            return;
        }

        // Position WorldRoot immediately so the scene appears right away, even
        // while the async anchor creation is in progress (takes 1–2 frames).
        worldRoot.SetPositionAndRotation(pos, rot);

        if (_creatingAnchor)
        {
            Debug.Log($"[WorldRootAnchorBridge] Ignoring pose (anchor creation in progress)");
            return;
        }

        // Wait for SLAM system to initialize before first anchor attempt.
        // SLAM needs 3-5 seconds after app start to build initial spatial map.
        if (_initStartTime >= 0f && _retryCount == 0 && Time.time - _initStartTime < initialDelayBeforeFirstAttempt)
        {
            float waitRemaining = initialDelayBeforeFirstAttempt - (Time.time - _initStartTime);
            Debug.Log($"[WorldRootAnchorBridge] Waiting {waitRemaining:F1}s for SLAM initialization before first anchor attempt…");
            return;
        }

        // Wait out the cooldown between retries — SLAM may need a few seconds
        // after app launch to re-localize against the stored room map.
        if (_retryAfterTime >= 0f && Time.time < _retryAfterTime)
        {
            float waitRemaining = _retryAfterTime - Time.time;
            Debug.Log($"[WorldRootAnchorBridge] Waiting {waitRemaining:F1}s before retry {_retryCount + 1}/{maxRetries}");
            return;
        }

        _creatingAnchor = true;

        // Do NOT call EraseAllAnchors here — nothing exists on first creation,
        // and the erase path tries cloud storage which times out in offline labs.
        // Erase is only called in ResetAnchor() when an anchor is known to exist.
        Debug.Log($"[WorldRootAnchorBridge] Creating spatial anchor (attempt {_retryCount + 1}/{maxRetries}) at pos={pos}, rot={rot.eulerAngles}…");
        anchorCore.InstantiateSpatialAnchor(null, pos, rot);
    }

    // ── Anchor callbacks ──────────────────────────────────────────────────────

    private void OnAnchorCreated(OVRSpatialAnchor anchor,
                                  OVRSpatialAnchor.OperationResult result)
    {
        _creatingAnchor = false;

        Debug.Log($"[WorldRootAnchorBridge] OnAnchorCreated callback: result={result} (int: {(int)result}), anchor={anchor}");

        if (result != OVRSpatialAnchor.OperationResult.Success || anchor == null)
        {
            _retryCount++;
            string anchorStatus = anchor == null ? "NULL" : $"created but result failed";
            Debug.LogWarning($"[WorldRootAnchorBridge] Anchor creation failed: result={result} ({(int)result}), anchor={anchorStatus}");

            if (_retryCount >= maxRetries)
            {
                _anchorsUnavailable = true;
                Debug.LogWarning($"[WorldRootAnchorBridge] Anchor creation failed {_retryCount} times " +
                                 $"(last error: {result}) — falling back to direct positioning permanently.");
            }
            else
            {
                _retryAfterTime = Time.time + retryDelaySecs;
                Debug.LogWarning($"[WorldRootAnchorBridge] Anchor creation failed ({result}), " +
                                 $"retry {_retryCount}/{maxRetries} in {retryDelaySecs}s.");
            }
            return;
        }

        Debug.Log($"[WorldRootAnchorBridge] Anchor creation SUCCEEDED: UUID={anchor.Uuid}");

        AttachToAnchor(anchor);

        PlayerPrefs.SetString(UuidPrefKey, anchor.Uuid.ToString());
        PlayerPrefs.Save();
        Debug.Log($"[WorldRootAnchorBridge] Anchor ready — UUID {anchor.Uuid} saved.");
    }

    private void OnAnchorsLoaded(List<OVRSpatialAnchor> anchors)
    {
        _loadingAnchor = false;

        Debug.Log($"[WorldRootAnchorBridge] OnAnchorsLoaded callback: anchors count={anchors?.Count ?? 0}");

        if (anchors == null || anchors.Count == 0)
        {
            Debug.LogWarning("[WorldRootAnchorBridge] Saved anchor not found (null or empty list) — " +
                             "will create a new one on next ArUco lock.");
            PlayerPrefs.DeleteKey(UuidPrefKey);
            PlayerPrefs.Save();
            return;
        }

        Debug.Log($"[WorldRootAnchorBridge] Successfully loaded {anchors.Count} anchor(s). Attaching to first anchor: {anchors[0].Uuid}");
        AttachToAnchor(anchors[0]);
        Debug.Log($"[WorldRootAnchorBridge] Anchor loaded from previous session: {anchors[0].Uuid}");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void TryLoadSavedAnchor()
    {
        if (!PlayerPrefs.HasKey(UuidPrefKey))
        {
            Debug.Log("[WorldRootAnchorBridge] No saved anchor UUID in PlayerPrefs — will create fresh anchor on first lock.");
            return;
        }

        var raw = PlayerPrefs.GetString(UuidPrefKey);
        if (!Guid.TryParse(raw, out var uuid))
        {
            Debug.LogWarning($"[WorldRootAnchorBridge] Saved UUID is invalid: '{raw}' — deleting and will create fresh anchor.");
            PlayerPrefs.DeleteKey(UuidPrefKey);
            return;
        }

        _loadingAnchor    = true;
        _loadingStartTime = Time.time;
        Debug.Log($"[WorldRootAnchorBridge] Attempting to load saved anchor {uuid} from previous session…");
        anchorCore.LoadAndInstantiateAnchors(null, new List<Guid> { uuid });
    }

    private void AttachToAnchor(OVRSpatialAnchor anchor)
    {
        Debug.Log($"[WorldRootAnchorBridge] AttachToAnchor: UUID={anchor.Uuid}, transform.position={anchor.transform.position}");

        _currentAnchor = anchor;
        worldRoot.SetParent(anchor.transform, worldPositionStays: false);
        worldRoot.localPosition = Vector3.zero;
        worldRoot.localRotation = Quaternion.identity;
        _anchorActive = true;

        Debug.Log($"[WorldRootAnchorBridge] ✅ ANCHOR ACTIVE: WorldRoot now parented to spatial anchor. Drift compensation engaged.");
    }
}
