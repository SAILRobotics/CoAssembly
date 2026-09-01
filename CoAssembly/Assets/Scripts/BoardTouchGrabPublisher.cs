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

    [Header("Touch volume")]
    [Tooltip("Invisible padding around the board used only to make hand contact reliable.")]
    [Min(0f)] public float interactionPadding = 0.01f;
    [Tooltip("Minimum front/back depth of the invisible touch volume.")]
    [Min(0.015f)] public float minimumInteractionDepth = 0.025f;

    public bool IsGrabbed => _selectCount > 0;

    private int   _selectCount;
    private float _nextPoseSendTime;
    private Behaviour[] _touchInteractables = System.Array.Empty<Behaviour>();

    public void SetInteractionEnabled(bool interactionEnabled)
    {
        if (!interactionEnabled)
            _selectCount = 0;

        // Enable dependencies before the interactable/listener; disable in
        // reverse. Leaving TouchHandGrabInteractable alive while Grabbable is
        // disabled can prevent ISDK from selecting it after a mode change.
        if (interactionEnabled)
        {
            if (pointable is Behaviour pointableBehaviour)
                pointableBehaviour.enabled = true;
            foreach (Behaviour touchInteractable in _touchInteractables)
                if (touchInteractable != null) touchInteractable.enabled = true;
            enabled = true;
        }
        else
        {
            foreach (Behaviour touchInteractable in _touchInteractables)
                if (touchInteractable != null) touchInteractable.enabled = false;
            if (pointable is Behaviour pointableBehaviour)
                pointableBehaviour.enabled = false;
            enabled = false;
        }
    }

    private void Awake()
    {
        if (board == null) board = transform;
        _touchInteractables = System.Array.FindAll(
            GetComponents<Behaviour>(), component =>
                component != null
                && component.GetType().Name == "TouchHandGrabInteractable");

        // The rendered board remains physically thin, but hand tracking needs
        // a forgiving contact volume to avoid tunnelling and tracking jitter.
        BoxCollider interactionCollider = GetComponent<BoxCollider>();
        if (interactionCollider != null)
        {
            Vector3 size = interactionCollider.size;
            // Keep this only slightly larger than the rendered board. A deep
            // TouchHandGrab volume continues overlapping an opened hand and
            // feels as though the board is stuck to it.
            float padding = Mathf.Max(0.01f, interactionPadding);
            size.x += 2f * padding;
            size.y += 2f * padding;
            size.z = Mathf.Max(size.z,
                Mathf.Max(0.025f, minimumInteractionDepth));
            interactionCollider.size = size;
        }
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
