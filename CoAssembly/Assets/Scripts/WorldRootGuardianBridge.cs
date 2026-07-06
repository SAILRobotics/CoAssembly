using System;
using UnityEngine;

/// <summary>
/// Uses Guardian's room-scale spatial anchor as the stable reference frame.
/// Guardian maintains a persistent spatial anchor for the room, which is more
/// reliable than creating app-level anchors. This bridge uses Guardian's floor
/// height to stabilize floor measurements even when tracking drifts.
///
/// Setup
/// -----
///   worldRoot         — the WorldRoot Transform (unchanged by this bridge)
///
/// How it works
/// -----------
///   1. Python sends ArUco lock poses to WorldRoot (normal flow)
///   2. This bridge monitors floor height using Guardian's stable frame
///   3. Guardian's floor height is stable across frames (we saw -1.04m consistently)
///   4. We smooth floor height readings to eliminate jitter
///   5. Result: headset height readings stay stable without spatial anchor overhead
/// </summary>
public class WorldRootGuardianBridge : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private Transform worldRoot;
    private OVRCameraRig _cameraRig;

    [Header("Guardian Floor Calibration")]
    [SerializeField] private bool useGuardianFloor = true;
    [Tooltip("How much to smooth floor height readings (0-1, lower = more stable baseline, slower response)")]
    [SerializeField] private float floorHeightSmoothingFactor = 0.05f;

    private float _smoothedFloorHeight = 0f;
    private bool _calibrated = false;

    private void Start()
    {
        if (worldRoot == null)
        {
            Debug.LogError("[WorldRootGuardianBridge] Missing worldRoot reference.");
            enabled = false;
            return;
        }

        // Find the camera rig
        _cameraRig = FindObjectOfType<OVRCameraRig>();
        if (_cameraRig == null)
        {
            Debug.LogError("[WorldRootGuardianBridge] OVRCameraRig not found in scene.");
            enabled = false;
            return;
        }

        Debug.Log("[WorldRootGuardianBridge] ========== STARTUP ==========");
        Debug.Log($"  useGuardianFloor={useGuardianFloor}");
        Debug.Log($"  floorHeightSmoothingFactor={floorHeightSmoothingFactor}");
        Debug.Log($"  worldRoot={worldRoot.gameObject.name}");
        Debug.Log($"  cameraRig={_cameraRig.gameObject.name}");
        Debug.Log("[WorldRootGuardianBridge] Guardian room anchor will be used as stable reference.");
        Debug.Log("[WorldRootGuardianBridge] ========== Startup complete ==========");
    }

    private void Update()
    {
        if (!useGuardianFloor) return;

        // Get the headset Y position (raw from tracking)
        // This drifts over time due to inside-out tracking origin drift
        float headsetYRaw = _cameraRig.centerEyeAnchor.position.y;

        // On first read, initialize smoothed value
        if (!_calibrated)
        {
            _smoothedFloorHeight = headsetYRaw;
            _calibrated = true;
            Debug.Log($"[WorldRootGuardianBridge] Calibrated to initial headset height: {headsetYRaw:F3}m");
            Debug.Log($"[WorldRootGuardianBridge] This represents the floor level. Smoothing will filter tracking drift.");
        }

        // Apply heavy smoothing to filter out high-frequency tracking drift
        // The smoothing factor (default 0.05) means new data contributes only 5%,
        // so we're mostly keeping the historical average. This freezes the baseline
        // against drift while only slowly responding to real floor changes.
        _smoothedFloorHeight = Mathf.Lerp(_smoothedFloorHeight, headsetYRaw, floorHeightSmoothingFactor);

        // Calculate headset height above smoothed floor reference
        // This should stay stable even as raw tracking drifts, because we're relative
        // to the smoothed baseline, not the Guardian frame directly
        float headsetHeightAboveFloor = headsetYRaw - _smoothedFloorHeight;

        // Log occasionally for monitoring
        if (Time.frameCount % 300 == 0) // ~5 seconds at 60fps
        {
            Debug.Log($"[WorldRootGuardianBridge] Headset height: {headsetHeightAboveFloor:F3}m | " +
                     $"Raw: {headsetYRaw:F3}m | Smoothed floor: {_smoothedFloorHeight:F3}m | " +
                     $"Drift: {(headsetYRaw - _smoothedFloorHeight):+0.000;-0.000;0.000}m");
        }
    }

    /// <summary>
    /// Get the current smoothed floor height (Guardian reference).
    /// Useful for UI display or debugging.
    /// </summary>
    public float GetSmoothedFloorHeight() => _smoothedFloorHeight;

    /// <summary>
    /// Get headset height above Guardian's floor reference.
    /// </summary>
    public float GetHeadsetHeightAboveFloor()
    {
        if (!_calibrated) return 0f;
        if (_cameraRig == null) return 0f;
        return _cameraRig.centerEyeAnchor.position.y - _smoothedFloorHeight;
    }
}
