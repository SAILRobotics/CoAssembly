using UnityEngine;

/// <summary>
/// Attach to the "Relock Button" GameObject (child of WorldRoot).
/// Bridges the ISDK select event to ToolClickPublisher so Python receives a
/// "selected" message and can trigger a proximity relock.
///
/// Requires on the same GameObject:
///   - ToolClickPublisher  (Tool Id = world marker ArUco ID, e.g. 100)
///   - ToolColorReceiver   (Tool Id = world marker ArUco ID, e.g. 100)
///
/// Do NOT add or modify the BoxCollider here — it is already configured in the
/// scene and referenced by ColliderSurface for ISDK ray interaction.
///
/// Scene wiring:
///   PointableUnityEventWrapper → WhenSelect → WorldMarkerInteractable.OnInteractorSelect
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

    /// <summary>
    /// Wire this to PointableUnityEventWrapper.WhenSelect.
    /// Sends a "selected" event to Python, which triggers the relock if conditions are met.
    /// </summary>
    public void OnInteractorSelect()
    {
        _publisher.OnSelected();
    }
}
