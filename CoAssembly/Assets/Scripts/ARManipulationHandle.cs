using UnityEngine;
using Oculus.Interaction;
using Oculus.Interaction.HandGrab;

/// <summary>
/// Full-sized AR manipulation handle.
///
/// Inspector setup:
///   mirroredGrabInteractable — drag the mirrored HandGrabInteractable here
///   arBox, targetPublisher, worldRoot — as before
///
/// ISDK snaps the hand visual to the handle but does NOT move the handle object.
/// We track interactor.transform (which follows the physical hand in the rig)
/// to compute the delta and apply it to the box.
/// </summary>
[RequireComponent(typeof(HandGrabInteractable))]
public class ARManipulationHandle : MonoBehaviour
{
    [Header("ISDK — drag mirrored interactable here")]
    public HandGrabInteractable mirroredGrabInteractable;

    [Header("AR references")]
    public GameObject          arBox;
    public GameObject          arHandle;
    public TargetPosePublisher targetPublisher;
    public Transform           worldRoot;
    [Tooltip("Physical HalfBoard thickness in metres.")]
    public float               boardThickness = 0.015f;
    [Tooltip("Full Handle.gltf depth along its local X axis, in metres.")]
    public float               handleDepth = 0.0210058f;
    [Tooltip("Pose stream rate while the AR board is being dragged.")]
    [Range(5f, 60f)] public float poseStreamHz = 20f;
    public bool IsGrabbed => _isGrabbed;
    public float HandleHalfDepth => handleDepth * 0.5f;

    private HandGrabInteractable _grabInteractable;
    private HandGrabInteractor   _currentInteractor;
    private bool       _isGrabbed;

    private Vector3    _grabHandPos;   // interactor world pos at grab time
    private Quaternion _grabHandRot;   // interactor world rot at grab time
    private Vector3    _grabBoxPos;    // box world pos at grab time
    private Quaternion _grabBoxRot;    // box world rot at grab time
    private float       _nextPoseSendTime;

    private void PublishPose(string manipulationState)
    {
        if (targetPublisher == null || worldRoot == null || arBox == null) return;
        Vector3 localPos = worldRoot.InverseTransformPoint(arBox.transform.position);
        Quaternion localRot = Quaternion.Inverse(worldRoot.rotation) * arBox.transform.rotation;
        targetPublisher.SendPose(localPos, localRot, manipulationState);
    }

    /// <summary>End an active grab without publishing another robot target.</summary>
    public void CancelManipulation()
    {
        _isGrabbed = false;
        _currentInteractor = null;
    }

    private void Awake()
    {
        _grabInteractable = GetComponent<HandGrabInteractable>();

    }

    private void OnEnable()
    {
        _grabInteractable.WhenSelectingInteractorAdded.Action   += OnGrabbed;
        _grabInteractable.WhenSelectingInteractorRemoved.Action += OnReleased;
        if (mirroredGrabInteractable != null)
        {
            mirroredGrabInteractable.WhenSelectingInteractorAdded.Action   += OnGrabbed;
            mirroredGrabInteractable.WhenSelectingInteractorRemoved.Action += OnReleased;
        }
    }

    private void OnDisable()
    {
        _grabInteractable.WhenSelectingInteractorAdded.Action   -= OnGrabbed;
        _grabInteractable.WhenSelectingInteractorRemoved.Action -= OnReleased;
        if (mirroredGrabInteractable != null)
        {
            mirroredGrabInteractable.WhenSelectingInteractorAdded.Action   -= OnGrabbed;
            mirroredGrabInteractable.WhenSelectingInteractorRemoved.Action -= OnReleased;
        }
    }

    private bool TryGetHandPose(HandGrabInteractor interactor, out Pose pose)
    {
        if (interactor != null && interactor.Hand.GetRootPose(out pose))
            return true;
        pose = new Pose(transform.position, transform.rotation);
        return false;
    }

    private void OnGrabbed(HandGrabInteractor interactor)
    {
        if (_isGrabbed || arBox == null) return;
        _isGrabbed         = true;
        _currentInteractor = interactor;

        TryGetHandPose(interactor, out Pose handPose);
        _grabHandPos = handPose.position;
        _grabHandRot = handPose.rotation;
        _grabBoxPos  = arBox.transform.position;
        _grabBoxRot  = arBox.transform.rotation;

        arBox.SetActive(true);
        _nextPoseSendTime = Time.unscaledTime;
        PublishPose("grabbed");
        Debug.Log($"[ARManipulationHandle] Grabbed — hand at {handPose.position}");
    }

    private void OnReleased(HandGrabInteractor interactor)
    {
        if (!_isGrabbed || arBox == null) return;
        _isGrabbed         = false;
        _currentInteractor = null;
        // The released board is the desired target visualization. Keep it
        // visible until GripStateReceiver confirms the live board reached it.
        arBox.SetActive(true);

        PublishPose("released");

        Debug.Log("[ARManipulationHandle] Released — target pose sent");
    }

    private void Update()
    {
        if (!_isGrabbed || _currentInteractor == null || arBox == null) return;

        if (!TryGetHandPose(_currentInteractor, out Pose currentHand)) return;

        Quaternion deltaRot = currentHand.rotation * Quaternion.Inverse(_grabHandRot);
        arBox.transform.position = currentHand.position + deltaRot * (_grabBoxPos - _grabHandPos);
        arBox.transform.rotation = deltaRot * _grabBoxRot;

        if (Time.unscaledTime >= _nextPoseSendTime)
        {
            _nextPoseSendTime = Time.unscaledTime + 1f / Mathf.Max(1f, poseStreamHz);
            PublishPose("dragging");
        }

        // Keep the Unity handle at the board-local offset and rotate it
        // 180 degrees about its own local X axis.
        if (arHandle != null)
        {
            arHandle.transform.position = arBox.transform.position
                - arBox.transform.rotation * Vector3.right * 0.0075f
                - arBox.transform.rotation * Vector3.forward * 0.2000f;
            arHandle.transform.rotation = arBox.transform.rotation
                * Quaternion.AngleAxis(180f, Vector3.right);
        }
    }
}
