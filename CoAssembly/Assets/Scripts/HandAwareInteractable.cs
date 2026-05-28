using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Add to any object needing hand-aware click events (e.g. TCPMarker).
/// Subscribes to the ISDK PointableUnityEventWrapper in code and sends
/// the triggering hand ("left" / "right") to Python via ToolClickPublisher.
///
/// Inspector setup:
///   _eventWrapper    — PointableUnityEventWrapper on this GameObject
///   _leftInteractor  — HandRayInteractor under LeftInteractions  > ... > HandRayInteractor
///   _rightInteractor — HandRayInteractor under RightInteractions > ... > HandRayInteractor
///
/// No manual event wiring in the Inspector needed — this script subscribes in code.
/// WorldMarkerInteractable is NOT required on this GameObject.
/// </summary>
[RequireComponent(typeof(ToolClickPublisher))]
[RequireComponent(typeof(ToolColorReceiver))]
public class HandAwareInteractable : MonoBehaviour
{
    [SerializeField] private PointableUnityEventWrapper _eventWrapper;
    [SerializeField] private RayInteractor              _leftInteractor;
    [SerializeField] private RayInteractor              _rightInteractor;

    private ToolClickPublisher _publisher;
    private string             _hoveringHand;

    private void Awake()
    {
        _publisher = GetComponent<ToolClickPublisher>();
    }

    private void OnEnable()
    {
        if (_eventWrapper == null) return;
        _eventWrapper.WhenSelect.AddListener(OnSelect);
        _eventWrapper.WhenHover.AddListener(OnHover);
        _eventWrapper.WhenUnhover.AddListener(OnUnhover);
    }

    private void OnDisable()
    {
        if (_eventWrapper == null) return;
        _eventWrapper.WhenSelect.RemoveListener(OnSelect);
        _eventWrapper.WhenHover.RemoveListener(OnHover);
        _eventWrapper.WhenUnhover.RemoveListener(OnUnhover);
    }

    private void OnSelect(PointerEvent evt)
    {
        string hand = IsLeft(evt.Identifier) ? "left" : "right";
        Debug.Log($"[HandAwareInteractable] {gameObject.name} clicked with {hand} hand (identifier={evt.Identifier})");
        _publisher.SendHandEvent("selected", hand);
    }

    private void OnHover(PointerEvent evt)
    {
        _hoveringHand = IsLeft(evt.Identifier) ? "left" : "right";
        _publisher.SendHandEvent("hover_enter", _hoveringHand);
    }

    private void OnUnhover(PointerEvent evt)
    {
        _publisher.SendHandEvent("hover_exit", _hoveringHand ?? "unknown");
        _hoveringHand = null;
    }

    private bool IsLeft(int identifier) =>
        _leftInteractor != null && _leftInteractor.Identifier == identifier;
}
