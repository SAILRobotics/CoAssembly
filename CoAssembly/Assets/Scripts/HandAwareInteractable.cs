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
/// Long press (opt-in via _enableLongPress, e.g. on the robot gripper / tool 200):
///   A press is classified when it ends instead of when it starts. Releasing before
///   _longPressSeconds sends "selected" on release; holding past the threshold sends
///   "long_press" immediately (while still held) and suppresses the "selected".
///   Cancelled presses send nothing. Objects without the flag keep the original
///   behaviour of sending "selected" as soon as the press begins.
///
/// No manual event wiring in the Inspector needed — this script subscribes in code.
/// WorldMarkerInteractable is NOT required on this GameObject.
/// </summary>
[RequireComponent(typeof(ToolClickPublisher))]
public class HandAwareInteractable : MonoBehaviour
{
    [SerializeField] private PointableUnityEventWrapper _eventWrapper;
    [SerializeField] private PointableUnityEventWrapper _pokeEventWrapper;

    [SerializeField] private bool  _enableLongPress  = false;
    [SerializeField] private float _longPressSeconds = 0.8f;
    // Hold-progress feedback: after a short dead zone (so ordinary taps barely
    // flicker) the object's color fills toward _longPressColor, reaching it at
    // the threshold and staying there until the press ends.
    [SerializeField] private float _feedbackDelaySeconds = 0.15f;
    [SerializeField] private Color _longPressColor = new Color(1f, 1f, 1f, 0.9f);

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

    // In-progress press (long-press mode only). One press is tracked at a time;
    // a second interactor selecting mid-press is ignored until the first ends.
    private bool   _pressActive;
    private int    _pressIdentifier;
    private string _pressHand;
    private float  _pressStartTime;
    private bool   _longPressSent;

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
        EndPress();
    }

    private void Update()
    {
        if (!_pressActive || _longPressSent) return;
        float elapsed = Time.time - _pressStartTime;
        float fill = (elapsed - _feedbackDelaySeconds)
                     / Mathf.Max(_longPressSeconds - _feedbackDelaySeconds, 0.01f);
        if (fill > 0f)
            ToolColorReceiver.ForTool(_publisher.toolId)?.SetOverlay(_longPressColor, fill);
        if (elapsed < _longPressSeconds) return;
        _longPressSent = true;
        Debug.Log($"[HandAwareInteractable] {gameObject.name} long-pressed with {_pressHand} hand");
        _publisher.SendHandEvent("long_press", _pressHand);
    }

    private void EndPress()
    {
        if (!_pressActive) return;
        _pressActive = false;
        if (_publisher != null)
            ToolColorReceiver.ForTool(_publisher.toolId)?.ClearOverlay();
    }

    private void SubscribeRay(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.AddListener(OnRaySelect);
        wrapper.WhenUnselect.AddListener(OnPressEnd);
        wrapper.WhenCancel.AddListener(OnPressCancel);
        wrapper.WhenHover.AddListener(OnRayHover);
        wrapper.WhenUnhover.AddListener(OnRayUnhover);
    }

    private void UnsubscribeRay(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.RemoveListener(OnRaySelect);
        wrapper.WhenUnselect.RemoveListener(OnPressEnd);
        wrapper.WhenCancel.RemoveListener(OnPressCancel);
        wrapper.WhenHover.RemoveListener(OnRayHover);
        wrapper.WhenUnhover.RemoveListener(OnRayUnhover);
    }

    private void SubscribePoke(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.AddListener(OnPokeSelect);
        wrapper.WhenUnselect.AddListener(OnPressEnd);
        wrapper.WhenCancel.AddListener(OnPressCancel);
        wrapper.WhenHover.AddListener(OnPokeHover);
        wrapper.WhenUnhover.AddListener(OnPokeUnhover);
    }

    private void UnsubscribePoke(PointableUnityEventWrapper wrapper)
    {
        if (wrapper == null) return;
        wrapper.WhenSelect.RemoveListener(OnPokeSelect);
        wrapper.WhenUnselect.RemoveListener(OnPressEnd);
        wrapper.WhenCancel.RemoveListener(OnPressCancel);
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
        if (_enableLongPress)
        {
            // Classify on release (OnPressEnd) or on the hold threshold (Update).
            if (_pressActive) return;
            _pressActive     = true;
            _pressIdentifier = evt.Identifier;
            _pressHand       = hand;
            _pressStartTime  = Time.time;
            _longPressSent   = false;
            return;
        }
        Debug.Log($"[HandAwareInteractable] {gameObject.name} clicked with {hand} hand (identifier={evt.Identifier})");
        _publisher.SendHandEvent("selected", hand);
    }

    private void OnPressEnd(PointerEvent evt)
    {
        if (!_pressActive || evt.Identifier != _pressIdentifier) return;
        EndPress();
        if (_longPressSent) return;
        Debug.Log($"[HandAwareInteractable] {gameObject.name} clicked with {_pressHand} hand (identifier={evt.Identifier})");
        _publisher.SendHandEvent("selected", _pressHand);
    }

    private void OnPressCancel(PointerEvent evt)
    {
        if (_pressActive && evt.Identifier == _pressIdentifier)
            EndPress();
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
