using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Bridges an ISDK PointableUnityEventWrapper select event to GearboxClickPublisher.
/// Add to any gearbox part (or the checkbox / reset object) that should publish a click
/// to Python. A lean version of HandAwareInteractable: it subscribes to WhenSelect in
/// code (no Inspector event wiring) but deliberately does NOT pull in ToolColorReceiver
/// (the gearbox parts are color-managed by GearboxCommandReceiver) and does not track
/// which hand clicked (the gearbox feature doesn't need it).
///
/// Inspector setup:
///   _eventWrapper — PointableUnityEventWrapper on this GameObject.
/// </summary>
[RequireComponent(typeof(GearboxClickPublisher))]
public class GearboxPartInteractable : MonoBehaviour
{
    [SerializeField] private PointableUnityEventWrapper _eventWrapper;

    private GearboxClickPublisher _publisher;

    private void Awake()
    {
        _publisher = GetComponent<GearboxClickPublisher>();
    }

    private void OnEnable()
    {
        if (_eventWrapper == null)
        {
            Debug.LogWarning($"[GearboxPartInteractable] {gameObject.name}: no PointableUnityEventWrapper assigned.");
            return;
        }
        _eventWrapper.WhenSelect.AddListener(OnSelect);
    }

    private void OnDisable()
    {
        if (_eventWrapper == null) return;
        _eventWrapper.WhenSelect.RemoveListener(OnSelect);
    }

    private void OnSelect(PointerEvent evt)
    {
        _publisher.SendSelected();
    }
}
