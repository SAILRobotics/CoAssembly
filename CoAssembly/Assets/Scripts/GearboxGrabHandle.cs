using UnityEngine;
using Oculus.Interaction.HandGrab;

/// <summary>
/// Makes an object grabbable/movable in XR: grab it with a hand (ISDK HandGrabInteractable) and it
/// moves + rotates 1:1 with your hand, so the gearbox can be repositioned and viewed from any angle.
///
/// Mirrors the proven hand-delta math in ARManipulationHandle: ISDK only tells us WHEN a grab
/// starts/ends (and snaps the hand visual); this script does the actual 6-DOF move by tracking the
/// interactor's hand-pose delta and applying it to `target`. There is no physics — the object stays
/// exactly where you release it. One hand at a time (the first grab wins), giving free move+rotate.
///
/// Setup (or use Tools > Gearbox > Make Gearbox Grabbable):
///   • a HandGrabInteractable + a kinematic Rigidbody on the object (the interactable grabs the
///     object's child colliders, so the existing per-part click colliders double as the grab volume);
///   • assign that interactable to `grabInteractable` and set `target` to the transform to move
///     (the gearbox root — the default is this object's transform).
///
/// Grab (hand) and click (ray-select on a part) use different interactors, so moving the whole
/// gearbox never conflicts with clicking a part to run its assembly stage.
/// </summary>
public class GearboxGrabHandle : MonoBehaviour
{
    [Tooltip("ISDK hand-grab interactable that fires the grab/release events. " +
             "Defaults to one on this GameObject.")]
    [SerializeField] private HandGrabInteractable grabInteractable;

    [Tooltip("Transform to move + rotate while grabbed. Defaults to this object's transform " +
             "(set it to the gearbox root).")]
    [SerializeField] private Transform target;

    private HandGrabInteractor _interactor;
    private bool       _grabbed;
    private Vector3    _grabHandPos;    // interactor world pos at grab time
    private Quaternion _grabHandRot;    // interactor world rot at grab time
    private Vector3    _grabTargetPos;  // target world pos at grab time
    private Quaternion _grabTargetRot;  // target world rot at grab time

    private void Awake()
    {
        if (grabInteractable == null) grabInteractable = GetComponent<HandGrabInteractable>();
        if (target == null)           target = transform;
    }

    private void OnEnable()
    {
        if (grabInteractable == null) return;
        grabInteractable.WhenSelectingInteractorAdded.Action   += OnGrabbed;
        grabInteractable.WhenSelectingInteractorRemoved.Action += OnReleased;
    }

    private void OnDisable()
    {
        if (grabInteractable == null) return;
        grabInteractable.WhenSelectingInteractorAdded.Action   -= OnGrabbed;
        grabInteractable.WhenSelectingInteractorRemoved.Action -= OnReleased;
    }

    private bool TryGetHandPose(HandGrabInteractor interactor, out Pose pose)
    {
        if (interactor != null && interactor.Hand != null && interactor.Hand.GetRootPose(out pose))
            return true;
        pose = new Pose(target.position, target.rotation);
        return false;
    }

    private void OnGrabbed(HandGrabInteractor interactor)
    {
        if (_grabbed) return;                               // first hand wins (one-hand 6-DOF)
        if (!TryGetHandPose(interactor, out Pose hand)) return;
        _grabbed       = true;
        _interactor    = interactor;
        _grabHandPos   = hand.position;
        _grabHandRot   = hand.rotation;
        _grabTargetPos = target.position;
        _grabTargetRot = target.rotation;
    }

    private void OnReleased(HandGrabInteractor interactor)
    {
        if (!_grabbed || interactor != _interactor) return;
        _grabbed    = false;
        _interactor = null;
    }

    private void Update()
    {
        if (!_grabbed || _interactor == null) return;
        if (!TryGetHandPose(_interactor, out Pose hand)) return;

        // Apply the hand's rotation + translation delta to the target (same formula as
        // ARManipulationHandle): rotate the grab-time offset by the hand's rotation delta.
        Quaternion deltaRot = hand.rotation * Quaternion.Inverse(_grabHandRot);
        target.position = hand.position + deltaRot * (_grabTargetPos - _grabHandPos);
        target.rotation = deltaRot * _grabTargetRot;
    }
}
