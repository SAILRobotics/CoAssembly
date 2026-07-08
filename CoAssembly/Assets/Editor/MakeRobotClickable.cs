using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Oculus.Interaction;
using Oculus.Interaction.Surfaces;

/// <summary>
/// Editor tool that makes the whole UR10e arm clickable, so clicking anywhere on
/// it triggers the same "go to the opposing hand" behaviour that the TCPMarker cube
/// does today.
///
/// It builds the exact ISDK ray-interaction stack TCPMarker uses onto each link's
/// visual mesh, all publishing the same tool_id (200 by default):
///
///   CapsuleCollider -> ColliderSurface -> RayInteractable ->
///   PointableUnityEventWrapper -> HandAwareInteractable -> ToolClickPublisher(toolId)
///
/// Python (main_with_robot.py) keys purely on tool_id == 200 + the reported hand, so
/// NO Python change is needed — every link just emits the identical click event.
///
/// Usage:
///   1. Tools > Robot > Make Arm Clickable
///   2. Drag in the Left / Right hand RayInteractors (so hand detection works).
///   3. Select the arm root (or the individual link GameObjects) in the Hierarchy.
///   4. "Set Up …". Save the scene.
///
/// References are assigned via SerializedObject using each component's real serialized
/// field name (_collider / _surface / _pointable / _eventWrapper) rather than the SDK's
/// Inject* methods, so it doesn't depend on a particular ISDK version's public API.
/// The operation is idempotent (skips anything that already has a ToolClickPublisher,
/// which also skips TCPMarker) and fully undoable.
/// </summary>
public class MakeRobotClickable : EditorWindow
{
    private RayInteractor _leftInteractor;
    private RayInteractor _rightInteractor;
    private int _toolId = 200;

    [MenuItem("Tools/Robot/Make Arm Clickable")]
    private static void Open() => GetWindow<MakeRobotClickable>("Make Arm Clickable");

    private void OnGUI()
    {
        EditorGUILayout.HelpBox(
            "Select the UR10e arm root (or individual link GameObjects) in the Hierarchy, " +
            "assign the two hand ray interactors, then Set Up.\n\n" +
            "Every visual mesh under the selection gets a capsule collider + the ISDK ray-" +
            "interaction stack that publishes tool_id " + _toolId + " on click — identical to " +
            "the TCPMarker cube, so clicking anywhere on the arm makes the robot go to the " +
            "opposing hand. No Python changes needed.",
            MessageType.Info);

        GUILayout.Space(4);
        _leftInteractor  = (RayInteractor)EditorGUILayout.ObjectField(
            "Left Interactor",  _leftInteractor,  typeof(RayInteractor), true);
        _rightInteractor = (RayInteractor)EditorGUILayout.ObjectField(
            "Right Interactor", _rightInteractor, typeof(RayInteractor), true);
        _toolId = EditorGUILayout.IntField("Tool Id", _toolId);

        if (_leftInteractor == null || _rightInteractor == null)
            EditorGUILayout.HelpBox(
                "Interactors not assigned — hand detection can't tell left from right, so every " +
                "click will report the same hand (the arm would always go to one side).",
                MessageType.Warning);

        GUILayout.Space(8);
        int n = Selection.gameObjects.Length;
        using (new EditorGUI.DisabledScope(n == 0))
        {
            if (GUILayout.Button($"Set Up {n} selected object(s) and their meshes"))
                Run(Selection.gameObjects);
        }

        GUILayout.Space(4);
        using (new EditorGUI.DisabledScope(n == 0))
        {
            if (GUILayout.Button("Assign interactors to selected HandAware components (fixes TCPMarker)"))
                PatchInteractors(Selection.gameObjects);
        }
    }

    private void Run(GameObject[] roots)
    {
        // Gather every visual mesh under the selection (children included).
        var targets = new List<MeshFilter>();
        foreach (var root in roots)
            targets.AddRange(root.GetComponentsInChildren<MeshFilter>(true));

        Undo.IncrementCurrentGroup();
        Undo.SetCurrentGroupName("Make Arm Clickable");
        int group = Undo.GetCurrentGroup();

        int done = 0, skipped = 0;
        foreach (var mf in targets)
        {
            switch (SetupOne(mf))
            {
                case Result.Added:   done++;    break;
                case Result.Skipped: skipped++;  break;
            }
        }

        Undo.CollapseUndoOperations(group);
        if (done > 0) EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log($"[MakeRobotClickable] Set up {done} mesh(es) with tool_id {_toolId}; skipped {skipped}. " +
                  "Save the scene to persist.");
    }

    private enum Result { Added, Skipped }

    private Result SetupOne(MeshFilter mf)
    {
        var go = mf.gameObject;

        if (mf.sharedMesh == null)
        {
            Debug.LogWarning($"[MakeRobotClickable] '{go.name}': MeshFilter has no mesh — skipped.");
            return Result.Skipped;
        }
        if (go.GetComponent<MeshRenderer>() == null)
        {
            Debug.LogWarning($"[MakeRobotClickable] '{go.name}': no MeshRenderer — skipped.");
            return Result.Skipped;
        }
        // Already interactive (or is TCPMarker) — leave it alone.
        if (go.GetComponent<ToolClickPublisher>() != null)
            return Result.Skipped;

        // 1. Capsule collider fitted to the mesh's local bounds.
        var cap = Undo.AddComponent<CapsuleCollider>(go);
        FitCapsule(cap, mf.sharedMesh);

        // 2. ColliderSurface wrapping that collider.
        var surf = Undo.AddComponent<ColliderSurface>(go);
        SetRef(surf, "_collider", cap);

        // 3. RayInteractable driven by that surface.
        var ray = Undo.AddComponent<RayInteractable>(go);
        SetRef(ray, "_surface", surf);

        // 4. PointableUnityEventWrapper exposing select/hover events.
        var wrap = Undo.AddComponent<PointableUnityEventWrapper>(go);
        SetRef(wrap, "_pointable", ray);

        // 5. ToolClickPublisher (required by HandAwareInteractable) — the thing that talks to Python.
        var pub = Undo.AddComponent<ToolClickPublisher>(go);
        SetInt(pub, "toolId", _toolId);

        // 5b. ToolColorReceiver (also required by HandAwareInteractable). Shares one socket across
        //     all instances; only one link per tool_id shows highlight tint, which is harmless here.
        var colr = Undo.AddComponent<ToolColorReceiver>(go);
        SetInt(colr, "toolId", _toolId);

        // 6. HandAwareInteractable — subscribes to the wrapper and reports the clicking hand.
        var hand = Undo.AddComponent<HandAwareInteractable>(go);
        SetRef(hand, "_eventWrapper", wrap);
        hand._leftInteractor  = _leftInteractor;
        hand._rightInteractor = _rightInteractor;

        EditorUtility.SetDirty(go);
        return Result.Added;
    }

    /// <summary>Assigns left/right interactors on every HandAwareInteractable in the selection
    /// (incl. children). Use this to repair TCPMarker, whose interactor slots are empty.</summary>
    private void PatchInteractors(GameObject[] roots)
    {
        int patched = 0;
        foreach (var root in roots)
        {
            foreach (var hand in root.GetComponentsInChildren<HandAwareInteractable>(true))
            {
                Undo.RecordObject(hand, "Assign hand interactors");
                hand._leftInteractor  = _leftInteractor;
                hand._rightInteractor = _rightInteractor;
                EditorUtility.SetDirty(hand);
                patched++;
            }
        }
        if (patched > 0) EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log($"[MakeRobotClickable] Assigned interactors on {patched} HandAwareInteractable(s).");
    }

    /// <summary>Fit a capsule to a mesh's local bounds: long axis = capsule direction,
    /// radius = half the larger of the two remaining extents.</summary>
    private static void FitCapsule(CapsuleCollider cap, Mesh mesh)
    {
        Bounds b = mesh.bounds;                 // local space, matches how the mesh renders
        Vector3 s = b.size;
        cap.center = b.center;

        int dir = 0; float longest = s.x;
        if (s.y > longest) { dir = 1; longest = s.y; }
        if (s.z > longest) { dir = 2; longest = s.z; }
        cap.direction = dir;                    // 0=X, 1=Y, 2=Z
        cap.height = longest;

        float radius = dir == 0 ? Mathf.Max(s.y, s.z)
                     : dir == 1 ? Mathf.Max(s.x, s.z)
                                : Mathf.Max(s.x, s.y);
        cap.radius = radius * 0.5f;
    }

    private static void SetRef(Object comp, string field, Object value)
    {
        var so = new SerializedObject(comp);
        var p = so.FindProperty(field);
        if (p == null)
        {
            Debug.LogError($"[MakeRobotClickable] Field '{field}' not found on {comp.GetType().Name} " +
                           "— the SDK's serialized name may have changed.");
            return;
        }
        p.objectReferenceValue = value;
        so.ApplyModifiedPropertiesWithoutUndo();
    }

    private static void SetInt(Object comp, string field, int value)
    {
        var so = new SerializedObject(comp);
        var p = so.FindProperty(field);
        if (p == null) { Debug.LogError($"[MakeRobotClickable] Field '{field}' not found on {comp.GetType().Name}."); return; }
        p.intValue = value;
        so.ApplyModifiedPropertiesWithoutUndo();
    }
}
