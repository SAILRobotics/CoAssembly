using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Oculus.Interaction;
using Oculus.Interaction.Surfaces;

/// <summary>
/// Editor tool that makes gearbox parts (and the checkbox / reset-X objects) clickable, so
/// clicking one publishes its GameObject NAME to Python via GearboxClickPublisher (port 5023).
///
/// It builds the same ISDK ray-interaction stack MakeRobotClickable uses, but with the gearbox's
/// lean publisher/interactable (no ToolColorReceiver — the gearbox parts are color-managed by
/// GearboxCommandReceiver) and a MeshCollider for precise, per-part click targets:
///
///   MeshCollider -> ColliderSurface -> RayInteractable ->
///   PointableUnityEventWrapper -> GearboxClickPublisher + GearboxPartInteractable
///
/// Usage:
///   1. Open a scene that has the ISDK interactor rig (e.g. Hand_MS.unity) and the gearbox.
///   2. Tools > Gearbox > Make Gearbox Clickable.
///   3a. Select the "Gearbox Assembly Named" root, then "Set up gearbox parts" — every mesh whose
///       name contains "Row" gets the stack.
///   3b. Create two small 3D objects, name them exactly "__checkbox__" and "__reset__", select
///       them, then "Set up selected as UI (checkbox / X)".
///   4. Assign those two objects (and the checkbox's Renderer) on the GearboxCommandReceiver.
///   5. Save the scene.
///
/// References are assigned via SerializedObject using each component's real serialized field name
/// (_collider / _surface / _pointable / _eventWrapper), so it doesn't depend on a particular ISDK
/// version's public API. Idempotent (skips anything that already has a GearboxClickPublisher).
/// </summary>
public class MakeGearboxClickable : EditorWindow
{
    // Port every GearboxClickPublisher this tool creates should bind. Must match
    // main_setting.GEARBOX_CLICK_PORT / gearbox_control.py's --click-port default.
    private const int ClickPort = 5023;

    [MenuItem("Tools/Gearbox/Make Gearbox Clickable")]
    private static void Open() => GetWindow<MakeGearboxClickable>("Make Gearbox Clickable");

    private void OnGUI()
    {
        EditorGUILayout.HelpBox(
            "Adds a MeshCollider + the ISDK ray-interaction stack (publishing the GameObject name " +
            $"on port {ClickPort}) to gearbox parts, so clicking a part isolates its row via Python.\n\n" +
            "• 'Set up gearbox parts' operates on every mesh under the selection whose name " +
            "contains \"Row\".\n" +
            "• 'Set up selected as UI' operates on exactly the selected objects — use it for the " +
            "checkbox, X, and revert button (name them \"__checkbox__\", \"__reset__\", " +
            "\"__revert__\"), then assign each on the GearboxCommandReceiver.\n" +
            $"• 'Update click port' rewrites every existing publisher under the selection to " +
            $"{ClickPort} — use it to migrate parts set up before the port changed.",
            MessageType.Info);

        GUILayout.Space(8);
        int n = Selection.gameObjects.Length;

        using (new EditorGUI.DisabledScope(n == 0))
        {
            if (GUILayout.Button($"Set up gearbox parts under {n} selected object(s)"))
                Run(Selection.gameObjects, onlyRowNamed: true);

            GUILayout.Space(4);
            if (GUILayout.Button($"Set up {n} selected object(s) as UI (checkbox / X / revert)"))
                Run(Selection.gameObjects, onlyRowNamed: false);

            GUILayout.Space(4);
            if (GUILayout.Button($"Update click port to {ClickPort} on selected + children"))
                SyncPort(Selection.gameObjects);
        }
    }

    // Rewrite the serialized `port` on every GearboxClickPublisher under the selection. The shared
    // socket binds whichever instance initializes first, so all must agree — this sets them all.
    private void SyncPort(GameObject[] roots)
    {
        Undo.IncrementCurrentGroup();
        Undo.SetCurrentGroupName("Update Gearbox Click Port");
        int group = Undo.GetCurrentGroup();

        int changed = 0;
        var seen = new HashSet<GearboxClickPublisher>();
        foreach (var root in roots)
            foreach (var pub in root.GetComponentsInChildren<GearboxClickPublisher>(true))
            {
                if (!seen.Add(pub)) continue;
                var so = new SerializedObject(pub);
                var p = so.FindProperty("port");
                if (p == null || p.intValue == ClickPort) continue;
                p.intValue = ClickPort;
                so.ApplyModifiedProperties();   // records undo
                EditorUtility.SetDirty(pub);
                changed++;
            }

        Undo.CollapseUndoOperations(group);
        if (changed > 0) EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log($"[MakeGearboxClickable] Set click port to {ClickPort} on {changed} publisher(s) " +
                  $"({seen.Count} found). Save the scene to persist.");
    }

    private void Run(GameObject[] roots, bool onlyRowNamed)
    {
        var targets = new List<MeshFilter>();
        foreach (var root in roots)
        {
            if (onlyRowNamed)
                targets.AddRange(root.GetComponentsInChildren<MeshFilter>(true));
            else
            {
                var mf = root.GetComponent<MeshFilter>();
                if (mf != null) targets.Add(mf);
            }
        }

        Undo.IncrementCurrentGroup();
        Undo.SetCurrentGroupName("Make Gearbox Clickable");
        int group = Undo.GetCurrentGroup();

        int done = 0, skipped = 0;
        foreach (var mf in targets)
        {
            if (onlyRowNamed && !mf.gameObject.name.Contains("Row")) { continue; }
            switch (SetupOne(mf))
            {
                case Result.Added:   done++;    break;
                case Result.Skipped: skipped++; break;
            }
        }

        Undo.CollapseUndoOperations(group);
        if (done > 0) EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log($"[MakeGearboxClickable] Set up {done} object(s); skipped {skipped}. Save the scene to persist.");
    }

    private enum Result { Added, Skipped }

    private Result SetupOne(MeshFilter mf)
    {
        var go = mf.gameObject;

        if (mf.sharedMesh == null)
        {
            Debug.LogWarning($"[MakeGearboxClickable] '{go.name}': MeshFilter has no mesh — skipped.");
            return Result.Skipped;
        }
        if (go.GetComponent<MeshRenderer>() == null)
        {
            Debug.LogWarning($"[MakeGearboxClickable] '{go.name}': no MeshRenderer — skipped.");
            return Result.Skipped;
        }
        if (go.GetComponent<GearboxClickPublisher>() != null)
            return Result.Skipped;   // already interactive

        // 1. MeshCollider (precise click target). Non-convex still supports Raycast.
        var col = go.GetComponent<Collider>();
        if (col == null)
        {
            var mc = Undo.AddComponent<MeshCollider>(go);
            mc.sharedMesh = mf.sharedMesh;
            col = mc;
        }

        // 2. ColliderSurface wrapping that collider.
        var surf = Undo.AddComponent<ColliderSurface>(go);
        SetRef(surf, "_collider", col);

        // 3. RayInteractable driven by that surface.
        var ray = Undo.AddComponent<RayInteractable>(go);
        SetRef(ray, "_surface", surf);

        // 4. PointableUnityEventWrapper exposing select/hover events.
        var wrap = Undo.AddComponent<PointableUnityEventWrapper>(go);
        SetRef(wrap, "_pointable", ray);

        // 5. GearboxClickPublisher — publishes this object's name to Python (required by the interactable).
        var pub = Undo.AddComponent<GearboxClickPublisher>(go);
        SetInt(pub, "port", ClickPort);   // stamp the port explicitly (don't rely on the code default)

        // 6. GearboxPartInteractable — subscribes to the wrapper's WhenSelect in code.
        var interact = Undo.AddComponent<GearboxPartInteractable>(go);
        SetRef(interact, "_eventWrapper", wrap);

        EditorUtility.SetDirty(go);
        return Result.Added;
    }

    private static void SetRef(Object comp, string field, Object value)
    {
        var so = new SerializedObject(comp);
        var p = so.FindProperty(field);
        if (p == null)
        {
            Debug.LogError($"[MakeGearboxClickable] Field '{field}' not found on {comp.GetType().Name} " +
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
        if (p == null)
        {
            Debug.LogError($"[MakeGearboxClickable] Field '{field}' not found on {comp.GetType().Name}.");
            return;
        }
        p.intValue = value;
        so.ApplyModifiedPropertiesWithoutUndo();
    }
}
