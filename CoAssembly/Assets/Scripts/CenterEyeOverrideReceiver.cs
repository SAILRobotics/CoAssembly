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

    [Header("Debug")]
    [SerializeField] private bool verboseLogs = false;

    // Last pose we applied — reapplied on Application.onBeforeRender (see below).
    private bool hasPose = false;
    private Vector3 lastPos;
    private Quaternion lastRot;

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

        lastPos = mCenter.GetColumn(3);
        lastRot = mCenter.rotation;
        hasPose = true;

        if (overrideRotation)
            centerEyeAnchor.SetPositionAndRotation(lastPos, lastRot);
        else
            centerEyeAnchor.position = lastPos;

        if (verboseLogs)
            Debug.Log($"[CenterEyeOverride] pos={lastPos} rot={(overrideRotation ? lastRot.eulerAngles : centerEyeAnchor.rotation.eulerAngles)}");
    }

    private void ReapplyBeforeRender()
    {
        if (!hasPose) return;

        if (overrideRotation)
            centerEyeAnchor.SetPositionAndRotation(lastPos, lastRot);
        else
            centerEyeAnchor.position = lastPos;
    }

    private void OnDestroy() => Application.onBeforeRender -= ReapplyBeforeRender;
    private void OnApplicationQuit() => Application.onBeforeRender -= ReapplyBeforeRender;
}
