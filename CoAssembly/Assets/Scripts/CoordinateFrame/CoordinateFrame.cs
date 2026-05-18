using UnityEngine;

public class CoordinateFrameGenerator : MonoBehaviour
{
    public float axisLength = 0.3f;
    public float axisRadius = 0.01f;

    void Start()
    {
        CreateAxis(Vector3.right, Color.red);   // X
        CreateAxis(Vector3.up, Color.green);    // Y
        CreateAxis(Vector3.forward, Color.blue); // Z
    }

    void CreateAxis(Vector3 direction, Color color)
    {
        GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        cylinder.transform.SetParent(transform);
        cylinder.transform.localScale = new Vector3(axisRadius, axisLength / 2, axisRadius);
        cylinder.transform.localPosition = direction * (axisLength / 2);
        cylinder.transform.rotation = Quaternion.FromToRotation(Vector3.up, direction);

        // Simple fix without creating new material
        cylinder.GetComponent<Renderer>().material.color = color;
    }
}