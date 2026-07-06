using UnityEngine;

public class FloorHeightReadout : MonoBehaviour
{
    public Transform centerEyeAnchor; // drag CenterEyeAnchor here in Inspector
    public bool logToConsole = true;

    void Update()
    {
        if (centerEyeAnchor == null) return;

        float heightAboveFloor = centerEyeAnchor.position.y;

        if (logToConsole)
        {
            Debug.Log($"Headset height above floor: {heightAboveFloor:F3} m");
        }
    }
}