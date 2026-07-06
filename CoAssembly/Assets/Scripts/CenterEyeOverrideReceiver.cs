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
///
/// Prediction: leftCamera.GetCameraPose() is time-aligned to
/// leftCamera.Timestamp — the passthrough camera's own (lower-rate, delayed)
/// image-capture time, not "now". During fast head motion that gap shows up
/// as visible lag. We estimate the head's linear/angular velocity by
/// finite-differencing consecutive camera frames, then extrapolate the
/// reconstructed center-eye pose forward by (now - leftCamera.Timestamp) to
/// approximately cancel that latency. This can slightly overshoot on sudden
/// stops/direction reversals (the usual predictive-tracking artifact), which
/// maxPredictionSeconds bounds.
/// </summary>
[DefaultExecutionOrder(10000)]   // run after OVRCameraRig so our position wins
public class CenterEyeOverrideReceiver : MonoBehaviour
{
    [Header("Target")]
    public Transform centerEyeAnchor;

    [Header("Left passthrough camera")]
    public PassthroughCameraAccess leftCamera;

    [Header("Latency compensation")]
    [SerializeField] private bool enablePrediction = true;
    [Tooltip("Upper bound on how far forward we'll extrapolate, in seconds. " +
             "Guards against overshoot if camera frames stall.")]
    [SerializeField] private float maxPredictionSeconds = 0.1f;

    [Header("Debug")]
    [SerializeField] private bool verboseLogs = false;

    // Last pose we applied — reapplied on Application.onBeforeRender (see below).
    private bool hasPose = false;
    private Vector3 lastPos;
    private Quaternion lastRot;

    // Finite-differenced velocity, updated whenever a new camera frame arrives.
    private bool hasPrevSample = false;
    private DateTime prevTimestamp;
    private Vector3 prevCenterPos;
    private Quaternion prevCenterRot = Quaternion.identity;
    private Vector3 linearVelocity;          // m/s
    private Vector3 angularAxis = Vector3.up; // unit axis
    private float angularSpeedDeg;            // deg/s

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
        DateTime timestamp = leftCamera.Timestamp;

        var mCam    = Matrix4x4.TRS(camPose.position, camPose.rotation, Vector3.one);
        var mOffset = Matrix4x4.TRS(lensOffset.position, lensOffset.rotation, Vector3.one);
        var mCenter = mCam * mOffset.inverse;

        Vector3 rawPos    = mCenter.GetColumn(3);
        Quaternion rawRot = mCenter.rotation;

        // A new camera frame arrived — refresh our velocity estimate via
        // finite difference against the previous frame's reconstructed pose.
        if (!hasPrevSample || timestamp != prevTimestamp)
        {
            if (hasPrevSample)
            {
                float dt = (float)(timestamp - prevTimestamp).TotalSeconds;
                if (dt > 1e-4f)
                {
                    linearVelocity = (rawPos - prevCenterPos) / dt;

                    Quaternion deltaRot = rawRot * Quaternion.Inverse(prevCenterRot);
                    deltaRot.ToAngleAxis(out float deltaAngleDeg, out Vector3 axis);
                    if (deltaAngleDeg > 180f) deltaAngleDeg -= 360f;   // shortest path
                    if (axis.sqrMagnitude > 1e-6f)
                    {
                        angularAxis      = axis;
                        angularSpeedDeg  = deltaAngleDeg / dt;
                    }
                    else
                    {
                        angularSpeedDeg = 0f;
                    }
                }
            }
            prevTimestamp  = timestamp;
            prevCenterPos  = rawPos;
            prevCenterRot  = rawRot;
            hasPrevSample  = true;
        }

        Vector3 predictedPos    = rawPos;
        Quaternion predictedRot = rawRot;
        if (enablePrediction)
        {
            float predictDt = Mathf.Clamp((float)(DateTime.UtcNow - timestamp).TotalSeconds,
                                          0f, maxPredictionSeconds);
            predictedPos = rawPos + linearVelocity * predictDt;
            predictedRot = Quaternion.AngleAxis(angularSpeedDeg * predictDt, angularAxis) * rawRot;
        }

        lastPos = predictedPos;
        lastRot = predictedRot;
        hasPose = true;
        centerEyeAnchor.SetPositionAndRotation(lastPos, lastRot);

        if (verboseLogs)
            Debug.Log($"[CenterEyeOverride] pos={lastPos} rot={lastRot.eulerAngles}");
    }

    private void ReapplyBeforeRender()
    {
        if (hasPose)
            centerEyeAnchor.SetPositionAndRotation(lastPos, lastRot);
    }

    private void OnDestroy() => Application.onBeforeRender -= ReapplyBeforeRender;
    private void OnApplicationQuit() => Application.onBeforeRender -= ReapplyBeforeRender;
}
