using UnityEngine;

/// <summary>
/// Attach to any interactable world-space object (relock button, TCP marker, etc).
/// Bridges ISDK pointer events to ToolClickPublisher so Python receives click/hover messages.
///
/// Requires on the same GameObject:
///   - ToolClickPublisher  (Tool Id = unique integer agreed with Python)
///   - ToolColorReceiver   (Tool Id = same integer)
///
/// Scene wiring (PointableUnityEventWrapper):
///   WhenSelect   → WorldMarkerInteractable.OnInteractorSelect
///   WhenHover    → WorldMarkerInteractable.OnInteractorHoverEnter
///   WhenUnhover  → WorldMarkerInteractable.OnInteractorHoverExit
/// </summary>
[RequireComponent(typeof(ToolClickPublisher))]
[RequireComponent(typeof(ToolColorReceiver))]
public class WorldMarkerInteractable : MonoBehaviour
{
    private ToolClickPublisher _publisher;

    private void Awake()
    {
        _publisher = GetComponent<ToolClickPublisher>();
    }

    public void OnInteractorSelect()     => _publisher.OnSelected();
    public void OnInteractorHoverEnter() => _publisher.OnHoverEnter();
    public void OnInteractorHoverExit()  => _publisher.OnHoverExit();
}
