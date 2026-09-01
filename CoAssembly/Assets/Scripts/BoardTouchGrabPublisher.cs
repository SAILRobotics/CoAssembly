using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Bridges an ISDK grab (Touch Hand Grab, Hand Grab, …) on the AR board itself to
/// the robot pose stream.
///
/// The board is moved by ISDK's own <see cref="Grabbable"/> (a PointableElement),
/// so this component only listens to that element's pointer events and republishes
/// the resulting board pose — in WorldRoot-local space — via <see cref="TargetPosePublisher"/>.
///
/// Inspector setup:
///   pointable        — the Grabbable / PointableElement on the board
///   targetPublisher  — same TargetPosePublisher the handle uses
///   worldRoot        — same WorldRoot transform the handle uses
///   board            — the transform whose pose is published (defaults to this transform)
/// </summary>
public class BoardTouchGrabPublisher : MonoBehaviour
{
    [Header("ISDK")]
    [Tooltip("Grabbable / PointableElement on the board that ISDK moves.")]
    public PointableElement pointable;

    [Header("AR references")]
    public TargetPosePublisher targetPublisher;
    public Transform           worldRoot;
    [Tooltip("Transform whose pose is streamed. Defaults to this GameObject.")]
    public Transform           board;

    [Tooltip("Pose stream rate while the board is being dragged.")]
    [Range(5f, 60f)] public float poseStreamHz = 20f;

    public bool IsGrabbed => _selectCount > 0;

    private int   _selectCount;
    private float _nextPoseSendTime;

    public void SetInteractionEnabled(bool interactionEnabled)
    {
        if (!interactionEnabled)
            _selectCount = 0;
        enabled = interactionEnabled;
        if (pointable is Behaviour pointableBehaviour)
            pointableBehaviour.enabled = interactionEnabled;
    }

    private void Awake()
    {
        if (board == null) board = transform;
    }

    private void OnEnable()
    {
        if (pointable != null)
            pointable.WhenPointerEventRaised += HandlePointerEvent;
    }

    private void OnDisable()
    {
        if (pointable != null)
            pointable.WhenPointerEventRaised -= HandlePointerEvent;
    }

    private void Update()
    {
        if (_selectCount > 0 && Time.unscaledTime >= _nextPoseSendTime)
        {
            _nextPoseSendTime = Time.unscaledTime + 1f / Mathf.Max(1f, poseStreamHz);
            PublishPose("dragging");
        }
    }

    private void HandlePointerEvent(PointerEvent evt)
    {
        switch (evt.Type)
        {
            case PointerEventType.Select:
                _selectCount++;
                if (_selectCount == 1)
                {
                    _nextPoseSendTime = Time.unscaledTime;
                    PublishPose("grabbed");
                }
                break;

            case PointerEventType.Unselect:
            case PointerEventType.Cancel:
                if (_selectCount > 0)
                {
                    _selectCount--;
                    if (_selectCount == 0)
                        PublishPose("released");
                }
                break;
        }
    }

    private void PublishPose(string manipulationState)
    {
        if (targetPublisher == null || worldRoot == null || board == null) return;
        Vector3    localPos = worldRoot.InverseTransformPoint(board.position);
        Quaternion localRot = Quaternion.Inverse(worldRoot.rotation) * board.rotation;
        targetPublisher.SendPose(localPos, localRot, manipulationState);
    }
}
