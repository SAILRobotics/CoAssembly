using UnityEngine;

public class AddTipCollidersSimple : MonoBehaviour
{
    public float tipRadius = 0.008f; // ~8 mm
    public bool addRigidbodies = true;

    static readonly string[] tipTokens = {
        "thumb_tip","index_tip","middle_tip","ring_tip","pinky_tip",
        "little_tip","finger_tip","_tip"
    };

    void Start()
    {
        int count = 0;
        foreach (Transform t in GetComponentsInChildren<Transform>(true))
        {
            string n = t.name.ToLower();
            bool isTip = false;
            foreach (var token in tipTokens) { if (n.Contains(token)) { isTip = true; break; } }
            if (!isTip) continue;

            var sc = t.GetComponent<SphereCollider>() ?? t.gameObject.AddComponent<SphereCollider>();
            sc.isTrigger = false;
            sc.radius = tipRadius;

            if (addRigidbodies)
            {
                var rb = t.GetComponent<Rigidbody>() ?? t.gameObject.AddComponent<Rigidbody>();
                rb.isKinematic = true;
                rb.useGravity = false;
                rb.interpolation = RigidbodyInterpolation.Interpolate;
                rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
            }
            count++;
        }
        Debug.Log($"✅ Added fingertip colliders: {count} on {gameObject.name}");
    }
}