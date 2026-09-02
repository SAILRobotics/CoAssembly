using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Add to any object needing hand-aware click events (e.g. TCPMarker).
/// Subscribes to one or more ISDK PointableUnityEventWrappers in code and sends
/// the triggering hand ("left" / "right") to Python via ToolClickPublisher.
///
/// Inspector setup:
///   _eventWrapper      — PointableUnityEventWrapper for the Ray interactable on this GameObject
///   _pokeEventWrapper  — PointableUnityEventWrapper for the Poke interactable, if this object
///                        also has one (e.g. a ClippedCylinderSurface/PokeInteractable child).
///                        Leave unassigned if the object is Ray-only.
///   _leftInteractor    — the hand-rig RAY interactor used by the left hand
///   _rightInteractor   — the hand-rig RAY interactor used by the right hand
///   _leftPokeInteractor  — the hand-rig POKE interactor used by the left hand
///   _rightPokeInteractor — the hand-rig POKE interactor used by the right hand
///
/// Ray and Poke events are identified against separate interactor pairs because a
/// RayInteractor and a PokeInteractor on the same hand have different Identifiers —
/// checking a poke-triggered event against the ray pair (or vice versa) would never
/// match and the hand would misattribute. _leftInteractor/_rightInteractor are typed
/// MonoBehaviour (cast to IInteractor at runtime) rather than a concrete interactor
/// type so existing scene/prefab assignments and ToolSpawner's runtime assignment
/// keep binding correctly (RayInteractor and PokeInteractor both IS-A MonoBehaviour).
///
/// No manual event wiring in the Inspector needed — this script subscribes in code.
/// WorldMarkerInteractable is NOT required on this GameObject.
/// </summary>
[RequireComponent(typeof(ToolClickPublisher))]
public class HandAwareInteractable : MonoBehaviour
{
    [SerializeField] private PointableUnityEventWrapper _eventWrapper;
    [SerializeField] private PointableUnityEventWrapper _pokeEventWrapper;

    [Interface(typeof(IInteractor))]
    public MonoBehaviour _leftInteractor;
    [Interface(typeof(IInteractor))]
    public MonoBehaviour _rightInteractor;

    [Interface(typeof(IInteractor))]
    public MonoBehaviour _leftPokeInteractor;
    [Interface(typeof(IInteractor))]
    public MonoBehaviour _rightPokeInteractor;

    private ToolClickPublisher _publisher;
    private string             _hoveringHand;

    private void Awake()
    {
        _publisher = GetComponent<ToolClickPublisher>();
    }

    private void OnEnable()
    {
        SubscribeRay(_eventWrapper);
        SubscribePoke(_pokeEventWrapper);
    }

    private void OnDisable()
    {
        UnsubscribeRay(_eventWrapper);
        UnsubscribePoke(_pokeEventWrapper);
    }

    private void SubscribeRay(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.AddListener(OnRaySelect);
        wrapper.WhenHover.AddListener(OnRayHover);
        wrapper.WhenUnhover.AddListener(OnRayUnhover);
    }

    private void UnsubscribeRay(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.RemoveListener(OnRaySelect);
        wrapper.WhenHover.RemoveListener(OnRayHover);
        wrapper.WhenUnhover.RemoveListener(OnRayUnhover);
    }

    private void SubscribePoke(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.AddListener(OnPokeSelect);
        wrapper.WhenHover.AddListener(OnPokeHover);
        wrapper.WhenUnhover.AddListener(OnPokeUnhover);
    }

    private void UnsubscribePoke(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.RemoveListener(OnPokeSelect);
        wrapper.WhenHover.RemoveListener(OnPokeHover);
        wrapper.WhenUnhover.RemoveListener(OnPokeUnhover);
    }

    private void OnRaySelect(PointerEvent evt)  => OnSelect(evt, _leftInteractor, _rightInteractor);
    private void OnRayHover(PointerEvent evt)   => OnHover(evt, _leftInteractor, _rightInteractor);
    private void OnRayUnhover(PointerEvent evt) => OnUnhover(evt);

    private void OnPokeSelect(PointerEvent evt)  => OnSelect(evt, _leftPokeInteractor, _rightPokeInteractor);
    private void OnPokeHover(PointerEvent evt)   => OnHover(evt, _leftPokeInteractor, _rightPokeInteractor);
    private void OnPokeUnhover(PointerEvent evt) => OnUnhover(evt);

    private void OnSelect(PointerEvent evt, MonoBehaviour left, MonoBehaviour right)
    {
        string hand = IsLeft(evt.Identifier, left) ? "left" : "right";
        Debug.Log($"[HandAwareInteractable] {gameObject.name} clicked with {hand} hand (identifier={evt.Identifier})");
        _publisher.SendHandEvent("selected", hand);
    }

    private void OnHover(PointerEvent evt, MonoBehaviour left, MonoBehaviour right)
    {
        _hoveringHand = IsLeft(evt.Identifier, left) ? "left" : "right";
        _publisher.SendHandEvent("hover_enter", _hoveringHand);
    }

    private void OnUnhover(PointerEvent evt)
    {
        _publisher.SendHandEvent("hover_exit", _hoveringHand ?? "unknown");
        _hoveringHand = null;
    }

    private bool IsLeft(int identifier, MonoBehaviour leftInteractor) =>
        leftInteractor is IInteractor left && left.Identifier == identifier;
}
