using UnityEngine;

/// <summary>
/// Shifts this GameObject's world position by CenterEyeOverrideReceiver.CenterEyeDelta
/// each frame. Attach to the root of a hand-tracking visual (e.g. Meta's "[BuildingBlock]
/// Hand Tracking left/right", positioned each frame by OVRSkeleton from native hand-tracking
/// data — a separate system from centerEyeAnchor, so it isn't touched by
/// CenterEyeOverrideReceiver's override on its own).
///
/// Runs in LateUpdate() — after OVRSkeleton/OVRMeshRenderer's Update() has positioned the
/// hand for this frame (Unity runs ALL Update() before ANY LateUpdate() in a frame,
/// regardless of execution order) — and after CenterEyeOverrideReceiver's own LateUpdate()
/// (execution order 10000) has computed CenterEyeDelta for this frame, guaranteed by our
/// own execution order of 10001.
/// </summary>
[DefaultExecutionOrder(10001)]
public class HandVisualCenterEyeCompensator : MonoBehaviour
{
    void LateUpdate()
    {
        transform.position += CenterEyeOverrideReceiver.CenterEyeDelta;
    }
}
