using System;
using UnityEngine;
using Meta.XR;

/// <summary>
/// Shifts OVRCameraRig's trackingSpace so that CenterEyeAnchor ends up at the true
/// center-eye pose reconstructed from the left passthrough camera's real-time pose,
/// computed entirely on-device — no round trip to Python, which used to add a full
/// network+processing cycle of latency and made the pose visibly jittery/laggy
/// relative to real head motion.
///
/// leftCamera.GetCameraPose() (a native OVRPlugin head-pose query, unaffected by
/// anything we write here) internally computes camPose = headPose ∘ LensOffset, where
/// LensOffset is the camera's fixed, factory-calibrated extrinsic relative to the head
/// (leftCamera.Intrinsics.LensOffset — see PassthroughCameraAccess.cs). We invert that
/// composition to get the desired world pose for CenterEyeAnchor: headPose = camPose ∘
/// inv(LensOffset). Then we solve for the trackingSpace pose that places
/// CenterEyeAnchor exactly there (see ApplyCorrection below).
///
/// WHY trackingSpace AND NOT CenterEyeAnchor DIRECTLY: an earlier version of this
/// component wrote straight to CenterEyeAnchor.position/rotation. On a real headset,
/// OVRCameraRig.UpdateAnchors() drives leftEyeAnchor/rightEyeAnchor (the actual stereo
/// render cameras) from independent native eye-tracking queries, NOT from
/// CenterEyeAnchor — so that override never touched the rendered view at all. Worse,
/// Meta's Interaction SDK (hand tracking, ray interactors, etc.) converts tracking-space
/// poses to world space via TrackingToWorldTransformerOVR, which uses
/// OVRCameraRig.trackingSpace — also completely independent of CenterEyeAnchor. Since
/// trackingSpace is the shared parent of CenterEyeAnchor, leftEyeAnchor, rightEyeAnchor,
/// and everything the Interaction SDK derives from tracking space, shifting IT corrects
/// the actual render cameras and all hand/ray tracking consistently in one place —
/// CenterEyeAnchor is just the reference point we use to compute how far to shift it.
///
/// By default (overrideRotation = false) only position is corrected — CenterEyeAnchor's
/// own (OVR-tracked) rotation is left alone, since orientation tracking is already
/// low-latency/high-rate via the gyro and doesn't need correcting. Check overrideRotation
/// to also rotate trackingSpace so CenterEyeAnchor's rotation matches the reconstructed
/// camera-relative pose too — note this rotates everything hanging off trackingSpace
/// (controllers, hands, boundary) around CenterEyeAnchor's position, not just the view.
///
/// An earlier version of this component instead auto-calibrated the offset at runtime by
/// comparing CenterEyeAnchor's current pose against GetCameraPose()'s result — but
/// GetCameraPose() is time-aligned to the camera's own (lower-rate) capture timestamp,
/// not "now", so any head motion between those two moments got baked into the "fixed"
/// offset as a large, bogus error. Using the SDK's already-known LensOffset sidesteps
/// that timing mismatch entirely — no calibration step, no staleness risk.
///
/// Applies the correction every LateUpdate() and again on Application.onBeforeRender as
/// defensive insurance, in case anything else (recentering, boundary system) touches
/// trackingSpace between LateUpdate and render — recomputed fresh each time from
/// whatever trackingSpace/CenterEyeAnchor's current actual state is, so it's
/// self-correcting either way.
///
/// Attach to any persistent GameObject. Drag OVRCameraRig's TrackingSpace into
/// trackingSpace, its CenterEyeAnchor into centerEyeAnchor, and the left
/// PassthroughCameraAccess (e.g. the one used by PassthroughCameraPublisher) into
/// leftCamera.
/// </summary>
[DefaultExecutionOrder(10000)]
public class CenterEyeOverrideReceiver : MonoBehaviour
{
    [Header("Target")]
    [Tooltip("OVRCameraRig's TrackingSpace — the shared parent we actually move.")]
    public Transform trackingSpace;
    [Tooltip("OVRCameraRig's CenterEyeAnchor — used as the reference point to compute " +
             "how far trackingSpace needs to shift; not written to directly.")]
    public Transform centerEyeAnchor;

    [Header("Left passthrough camera")]
    public PassthroughCameraAccess leftCamera;

    [Header("Rotation")]
    [Tooltip("If false (default), only position is corrected — CenterEyeAnchor's own " +
             "OVR-tracked rotation (and everything else's) is left alone. If true, " +
             "trackingSpace is also rotated so CenterEyeAnchor's rotation matches the " +
             "reconstructed camera-relative pose.")]
    [SerializeField] private bool overrideRotation = false;

    [Header("Position axes")]
    [Tooltip("If true, only the reconstructed Y (height) is applied — X/Z are left as " +
             "CenterEyeAnchor's own OVR-tracked position. Useful if only vertical alignment " +
             "with the passthrough camera matters and you'd rather not touch X/Z at all.")]
    [SerializeField] private bool overrideYOnly = false;

    [Header("Position smoothing (optional)")]
    [Tooltip("The reconstructed position only changes when a new passthrough camera frame " +
             "arrives (lower rate than render), so it holds steady then steps — this can " +
             "read as jitter. Enabling this eases toward the target with Vector3.SmoothDamp " +
             "instead of snapping straight to it, trading a little added lag for a smoother " +
             "motion. It never overshoots past real data (unlike velocity extrapolation), " +
             "but it is always a bit behind — tune positionSmoothTime for the balance.")]
    [SerializeField] private bool smoothPosition = false;
    [SerializeField] private float positionSmoothTime = 0.05f;   // seconds

    [Header("Debug")]
    [SerializeField] private bool verboseLogs = false;

    // Raw reconstructed target this frame (before smoothing) — desired WORLD pose for CenterEyeAnchor.
    private Vector3 targetPos;
    private Quaternion targetRot;

    private bool hasPose = false;
    private Vector3 appliedPos;       // possibly SmoothDamp-eased version of targetPos
    private Vector3 smoothVelocity;   // SmoothDamp's internal velocity state

    void Start()
    {
        if (trackingSpace == null)
        {
            Debug.LogError("[CenterEyeOverride] trackingSpace is not assigned.");
            enabled = false;
            return;
        }
        if (centerEyeAnchor == null)
        {
            Debug.LogError("[CenterEyeOverride] centerEyeAnchor is not assigned.");
            enabled = false;
            return;
        }
        if (leftCamera == null)
        {
            Debug.LogError("[CenterEyeOverride] leftCamera is not assigned.");
            enabled = false;
            return;
        }

        Application.onBeforeRender += ApplyCorrection;
    }

    void LateUpdate()
    {
        if (!leftCamera.enabled || !leftCamera.IsPlaying)
            return;

        Pose camPose    = leftCamera.GetCameraPose();
        Pose lensOffset = leftCamera.Intrinsics.LensOffset;

        var mCam    = Matrix4x4.TRS(camPose.position, camPose.rotation, Vector3.one);
        var mOffset = Matrix4x4.TRS(lensOffset.position, lensOffset.rotation, Vector3.one);
        var mCenter = mCam * mOffset.inverse;   // desired WORLD pose for CenterEyeAnchor

        targetPos = mCenter.GetColumn(3);
        targetRot = mCenter.rotation;

        if (smoothPosition && hasPose)
            appliedPos = Vector3.SmoothDamp(appliedPos, targetPos, ref smoothVelocity, positionSmoothTime);
        else
            appliedPos = targetPos;   // first frame, or smoothing disabled — snap straight to target

        hasPose = true;
        ApplyCorrection();

        if (verboseLogs)
            Debug.Log($"[CenterEyeOverride] trackingSpace pos={trackingSpace.position} " +
                      $"centerEye pos={centerEyeAnchor.position} target={appliedPos}");
    }

    /// <summary>
    /// Solves for the trackingSpace pose that places CenterEyeAnchor at
    /// (appliedPos, desired rotation), then applies it. Re-derives everything from
    /// trackingSpace/CenterEyeAnchor's CURRENT actual transforms each call, so calling
    /// this more than once per frame (LateUpdate + onBeforeRender) is safe and
    /// self-correcting rather than compounding.
    /// </summary>
    private void ApplyCorrection()
    {
        if (!hasPose) return;

        Vector3 desiredCenterPos = overrideYOnly
            ? new Vector3(centerEyeAnchor.position.x, appliedPos.y, centerEyeAnchor.position.z)
            : appliedPos;
        Quaternion desiredCenterRot = overrideRotation ? targetRot : centerEyeAnchor.rotation;

        // CenterEyeAnchor's pose relative to trackingSpace, as OVR set it this frame —
        // untouched by us, so this reflects the real tracked relationship.
        var mTrackingSpaceNow = Matrix4x4.TRS(trackingSpace.position, trackingSpace.rotation, Vector3.one);
        var mCenterEyeNow     = Matrix4x4.TRS(centerEyeAnchor.position, centerEyeAnchor.rotation, Vector3.one);
        var mLocal            = mTrackingSpaceNow.inverse * mCenterEyeNow;

        var mDesiredCenterEye = Matrix4x4.TRS(desiredCenterPos, desiredCenterRot, Vector3.one);
        var mNewTrackingSpace = mDesiredCenterEye * mLocal.inverse;

        trackingSpace.SetPositionAndRotation(mNewTrackingSpace.GetColumn(3), mNewTrackingSpace.rotation);
    }

    private void OnDestroy() => Application.onBeforeRender -= ApplyCorrection;
    private void OnApplicationQuit() => Application.onBeforeRender -= ApplyCorrection;
}
