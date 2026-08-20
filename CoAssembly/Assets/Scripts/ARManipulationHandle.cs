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
    public bool IsGrabbed => _isGrabbed;
    public float HandleHalfDepth { get; private set; }
    public float BoardHalfThickness => boardThickness * 0.5f;

    private HandGrabInteractable _grabInteractable;
    private HandGrabInteractor   _currentInteractor;
    private bool       _isGrabbed;

    private Vector3    _grabHandPos;   // interactor world pos at grab time
    private Quaternion _grabHandRot;   // interactor world rot at grab time
    private Vector3    _grabBoxPos;    // box world pos at grab time
    private Quaternion _grabBoxRot;    // box world rot at grab time

    private void Awake()
    {
        _grabInteractable = GetComponent<HandGrabInteractable>();

        // Measure the handle's half-depth from its mesh so no manual entry is needed
        if (arHandle != null)
        {
            var mf = arHandle.GetComponentInChildren<MeshFilter>();
            if (mf != null && mf.sharedMesh != null)
                HandleHalfDepth = mf.sharedMesh.bounds.extents.z * arHandle.transform.lossyScale.z;
        }
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
        Debug.Log($"[ARManipulationHandle] Grabbed — hand at {handPose.position}");
    }

    private void OnReleased(HandGrabInteractor interactor)
    {
        if (!_isGrabbed || arBox == null) return;
        _isGrabbed         = false;
        _currentInteractor = null;

        if (targetPublisher != null && worldRoot != null)
        {
            Vector3    localPos = worldRoot.InverseTransformPoint(arBox.transform.position);
            Quaternion localRot = Quaternion.Inverse(worldRoot.rotation) * arBox.transform.rotation;
            targetPublisher.SendPose(localPos, localRot);
        }

        Debug.Log("[ARManipulationHandle] Released — target pose sent");
    }

    private void Update()
    {
        if (!_isGrabbed || _currentInteractor == null || arBox == null) return;

        if (!TryGetHandPose(_currentInteractor, out Pose currentHand)) return;

        Quaternion deltaRot = currentHand.rotation * Quaternion.Inverse(_grabHandRot);
        arBox.transform.position = currentHand.position + deltaRot * (_grabBoxPos - _grabHandPos);
        arBox.transform.rotation = deltaRot * _grabBoxRot;

        // Pin the handle to the corrected HalfBoard mesh's near face. The
        // visual correction places board thickness on logical local X.
        if (arHandle != null)
        {
            Vector3 boardNormal = arBox.transform.rotation * Vector3.right;
            arHandle.transform.position = arBox.transform.position
                - boardNormal * (BoardHalfThickness + HandleHalfDepth);
            arHandle.transform.rotation = arBox.transform.rotation
                * Quaternion.AngleAxis(-90f, Vector3.up);
        }
    }
}
