using System;
using UnityEngine;
using Meta.XR;

/// <summary>
/// Drives CenterEyeAnchor to the true center-eye pose reconstructed from the
/// left passthrough camera's real-time pose, computed entirely on-device —
/// no round trip to Python, which used to add a full network+processing
/// cycle of latency and made the pose visibly jittery/laggy relative to real
/// head motion.
///
/// leftCamera.GetCameraPose() (a native OVRPlugin head-pose query, unaffected
/// by anything we write to CenterEyeAnchor) internally computes
/// camPose = headPose ∘ LensOffset, where LensOffset is the camera's fixed,
/// factory-calibrated extrinsic relative to the head
/// (leftCamera.Intrinsics.LensOffset — see PassthroughCameraAccess.cs).
/// We just invert that composition: headPose = camPose ∘ inv(LensOffset).
///
/// By default (overrideRotation = false) only the reconstructed POSITION is
/// applied — CenterEyeAnchor's own (OVR-tracked) rotation is left alone,
/// since orientation tracking is already low-latency/high-rate via the gyro
/// and doesn't need correcting; only translation needs to be pulled onto the
/// camera's physical position. Check overrideRotation to also apply the
/// reconstructed rotation if needed.
///
/// An earlier version of this component instead auto-calibrated the offset
/// at runtime by comparing CenterEyeAnchor's current pose against
/// GetCameraPose()'s result — but GetCameraPose() is time-aligned to the
/// camera's own (lower-rate) capture timestamp, not "now", so any head
/// motion between those two moments got baked into the "fixed" offset as a
/// large, bogus error. Using the SDK's already-known LensOffset sidesteps
/// that timing mismatch entirely — no calibration step, no staleness risk.
///
/// Applies the override every LateUpdate() — after OVR has done its own
/// tracking update — and again on Application.onBeforeRender, since
/// OVRCameraRig re-drives CenterEyeAnchor from real HMD tracking on that same
/// event (OVRCameraRig.OnBeforeRenderCallback) after all scripts'
/// LateUpdate() has run. Without the second reapply, the two writers race
/// and the anchor flickers between the override pose and the real tracked
/// pose every frame.
///
/// Attach to any persistent GameObject. Drag OVRCameraRig's CenterEyeAnchor
/// into centerEyeAnchor, and the left PassthroughCameraAccess (e.g. the one
/// used by PassthroughCameraPublisher) into leftCamera.
/// </summary>
[DefaultExecutionOrder(10000)]   // run after OVRCameraRig so our position wins
public class CenterEyeOverrideReceiver : MonoBehaviour
{
    [Header("Target")]
    public Transform centerEyeAnchor;

    [Header("Left passthrough camera")]
    public PassthroughCameraAccess leftCamera;

    [Header("Rotation")]
    [Tooltip("If false (default), only CenterEyeAnchor's position is overridden — its own " +
             "OVR-tracked rotation is left alone. If true, rotation is also overridden to " +
             "match the reconstructed camera-relative pose.")]
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

    // Raw reconstructed target this frame (before smoothing).
    private Vector3 targetPos;
    private Quaternion lastRot;

    // What we actually applied to centerEyeAnchor.position — reapplied on
    // Application.onBeforeRender (see below). When smoothPosition is on,
    // this is the SmoothDamp-eased value, not the raw target.
    private bool hasPose = false;
    private Vector3 appliedPos;
    private Vector3 smoothVelocity;   // SmoothDamp's internal velocity state

    void Start()
    {
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

        // OVRCameraRig re-drives centerEyeAnchor from real HMD tracking on
        // Application.onBeforeRender (fires after every script's LateUpdate),
        // which otherwise fights our override and produces a per-frame
        // flicker between the two poses. Subscribing here — after
        // OVRCameraRig's own Start() runs, guaranteed by our
        // [DefaultExecutionOrder(10000)] — reapplies our pose last so it
        // always wins right before render.
        Application.onBeforeRender += ReapplyBeforeRender;
    }

    void LateUpdate()
    {
        if (!leftCamera.enabled || !leftCamera.IsPlaying)
            return;

        Pose camPose     = leftCamera.GetCameraPose();
        Pose lensOffset  = leftCamera.Intrinsics.LensOffset;

        var mCam    = Matrix4x4.TRS(camPose.position, camPose.rotation, Vector3.one);
        var mOffset = Matrix4x4.TRS(lensOffset.position, lensOffset.rotation, Vector3.one);
        var mCenter = mCam * mOffset.inverse;

        targetPos = mCenter.GetColumn(3);
        lastRot   = mCenter.rotation;

        if (smoothPosition && hasPose)
            appliedPos = Vector3.SmoothDamp(appliedPos, targetPos, ref smoothVelocity, positionSmoothTime);
        else
            appliedPos = targetPos;   // first frame, or smoothing disabled — snap straight to target

        hasPose = true;
        ApplyToAnchor();

        if (verboseLogs)
            Debug.Log($"[CenterEyeOverride] pos={appliedPos} target={targetPos} rot={(overrideRotation ? lastRot.eulerAngles : centerEyeAnchor.rotation.eulerAngles)}");
    }

    private void ReapplyBeforeRender()
    {
        if (!hasPose) return;
        ApplyToAnchor();
    }

    private void ApplyToAnchor()
    {
        // CenterEyeAnchor's own OVR-tracked pose was already (re-)written earlier this
        // frame — by OVRCameraRig's Update()/LateUpdate() before ours, or by its
        // onBeforeRender callback before this one — so reading it here for the axes we
        // don't touch always reflects the current real-time tracked value, not a stale one.
        Vector3 posToApply = overrideYOnly
            ? new Vector3(centerEyeAnchor.position.x, appliedPos.y, centerEyeAnchor.position.z)
            : appliedPos;

        if (overrideRotation)
            centerEyeAnchor.SetPositionAndRotation(posToApply, lastRot);
        else
            centerEyeAnchor.position = posToApply;
    }

    private void OnDestroy() => Application.onBeforeRender -= ReapplyBeforeRender;
    private void OnApplicationQuit() => Application.onBeforeRender -= ReapplyBeforeRender;
}
