using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using UnityEngine;
using Oculus.Interaction.HandGrab;

/// <summary>
/// Editor tool: makes the selected gearbox root grabbable/movable in XR — grab it with a hand to
/// move + rotate it and view from any angle. Wires the mechanical parts and reuses a working
/// HandGrabInteractable so the grab config matches this project's ISDK version.
///
/// Adds to the selected object:
///   • a kinematic Rigidbody (stays where released; no gravity), so its child part colliders form
///     one compound grab body — no extra box collider (which would block the per-part click ray);
///   • a HandGrabInteractable — COPIED from an existing one in the scene when possible (else added
///     with defaults), with its rigidbody re-pointed at this object;
///   • GearboxGrabHandle, which does the 1:1 hand-follow move/rotate.
///
/// Usage: open the scene with the ISDK hand rig, select "Gearbox Assembly Named",
/// Tools > Gearbox > Make Gearbox Grabbable, click the button, then Save the scene.
/// </summary>
public class MakeGearboxGrabbable : EditorWindow
{
    [MenuItem("Tools/Gearbox/Make Gearbox Grabbable")]
    private static void Open() => GetWindow<MakeGearboxGrabbable>("Make Gearbox Grabbable");

    private void OnGUI()
    {
        EditorGUILayout.HelpBox(
            "Makes the selected object grabbable: grab it with a hand in XR to move + rotate it, so " +
            "you can view the gearbox from any angle. Clicking a part (ray-select) still runs its " +
            "assembly stage — grab (hand) and click (ray) use different interactors.\n\n" +
            "Adds a kinematic Rigidbody + a HandGrabInteractable + the GearboxGrabHandle script. " +
            "Select the gearbox root (\"Gearbox Assembly Named\") first.\n\n" +
            "The HandGrabInteractable is copied from an existing one in the scene when possible so it " +
            "matches your rig; otherwise verify its supported grab types and collider list.",
            MessageType.Info);

        GUILayout.Space(8);
        var go = Selection.activeGameObject;
        using (new EditorGUI.DisabledScope(go == null))
            if (GUILayout.Button(go != null ? $"Make '{go.name}' grabbable" : "Select an object first"))
                Setup(go);
    }

    private void Setup(GameObject go)
    {
        Undo.IncrementCurrentGroup();
        Undo.SetCurrentGroupName("Make Gearbox Grabbable");
        int group = Undo.GetCurrentGroup();

        // 1. Kinematic Rigidbody — stays put, no gravity; makes the child colliders one grab body.
        var rb = go.GetComponent<Rigidbody>();
        if (rb == null) rb = Undo.AddComponent<Rigidbody>(go);
        rb.isKinematic = true;
        rb.useGravity  = false;

        // 2. HandGrabInteractable — replicate a working one if present, else add with defaults.
        if (go.GetComponent<HandGrabInteractable>() == null)
        {
            var source = Object.FindObjectsOfType<HandGrabInteractable>(true)
                               .FirstOrDefault(h => h.gameObject != go);
            if (source != null && ComponentUtility.CopyComponent(source)
                               && ComponentUtility.PasteComponentAsNew(go))
                Debug.Log($"[MakeGearboxGrabbable] Copied HandGrabInteractable config from '{source.name}'. " +
                          "Verify its collider list references this gearbox's colliders.");
            else
            {
                Undo.AddComponent<HandGrabInteractable>(go);
                Debug.LogWarning("[MakeGearboxGrabbable] No existing HandGrabInteractable found to copy — " +
                                 "added one with defaults. Verify its supported grab types + collider list " +
                                 "against a working handle in your scene.");
            }
        }
        var hgi = go.GetComponent<HandGrabInteractable>();
        SetRef(hgi, "_rigidbody", rb);   // re-point the interactable at our kinematic body

        // 3. GearboxGrabHandle — the actual move/rotate.
        var handle = go.GetComponent<GearboxGrabHandle>();
        if (handle == null) handle = Undo.AddComponent<GearboxGrabHandle>(go);
        SetRef(handle, "grabInteractable", hgi);
        SetRef(handle, "target", go.transform);

        Undo.CollapseUndoOperations(group);
        EditorUtility.SetDirty(go);
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log($"[MakeGearboxGrabbable] '{go.name}' is now grabbable (move + rotate). Save the scene to persist.");
    }

    private static void SetRef(Object comp, string field, Object value)
    {
        var so = new SerializedObject(comp);
        var p = so.FindProperty(field);
        if (p == null)
        {
            Debug.LogError($"[MakeGearboxGrabbable] Field '{field}' not found on {comp.GetType().Name} " +
                           "— the SDK's serialized name may differ; assign it in the Inspector.");
            return;
        }
        p.objectReferenceValue = value;
        so.ApplyModifiedPropertiesWithoutUndo();
    }
}
