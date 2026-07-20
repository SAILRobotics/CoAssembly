using System.Text.RegularExpressions;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Editor tool that renames gearbox objects to an underscore style, for readability:
///   BearingRow3Left               -> Bearing_Row3_Left
///   GearRodRow1                   -> GearRod_Row1
///   occurrence of BearingRow3Left -> occurrence of Bearing_Row3_Left
///
/// It inserts an underscore before the "Row{N}" token and before the Left/Right side wherever
/// they appear in a name, so both the "occurrence of ..." wrappers and the mesh children under
/// them are handled in one pass. It is idempotent (running twice adds no extra underscores) and
/// fully undoable.
///
/// The gearbox scripts are underscore-agnostic (GearboxCommandReceiver / gearbox_control.py parse
/// names with underscores stripped), so renaming does not break row/state/assembly/click logic.
///
/// Usage:
///   1. Tools > Gearbox > Rename With Underscores.
///   2. Select the "Gearbox Assembly Named" root (or any objects) in the Hierarchy.
///      (If the gearbox is a prefab instance, open it in Prefab Mode first to rename in the prefab
///       itself rather than as per-instance overrides.)
///   3. Click "Rename". Save the scene / prefab.
/// </summary>
public class RenameGearboxUnderscores : EditorWindow
{
    [MenuItem("Tools/Gearbox/Rename With Underscores")]
    private static void Open() => GetWindow<RenameGearboxUnderscores>("Rename With Underscores");

    private void OnGUI()
    {
        EditorGUILayout.HelpBox(
            "Renames every object under the selection whose name contains \"Row\" to an underscore " +
            "style, e.g. BearingRow3Left -> Bearing_Row3_Left. Handles the \"occurrence of ...\" " +
            "wrappers and the meshes beneath them. Idempotent and undoable.\n\n" +
            "Tip: to change the names in the prefab itself (not as instance overrides), open the " +
            "gearbox prefab in Prefab Mode before running.",
            MessageType.Info);

        GUILayout.Space(6);
        int n = Selection.gameObjects.Length;

        using (new EditorGUI.DisabledScope(n == 0))
        {
            if (GUILayout.Button($"Rename under {n} selected object(s)"))
                Run(Selection.gameObjects);

            GUILayout.Space(4);
            if (GUILayout.Button("Preview (log only, no changes)"))
                Run(Selection.gameObjects, previewOnly: true);
        }
    }

    private static void Run(GameObject[] roots, bool previewOnly = false)
    {
        if (!previewOnly)
        {
            Undo.IncrementCurrentGroup();
            Undo.SetCurrentGroupName("Rename Gearbox With Underscores");
        }
        int group = Undo.GetCurrentGroup();

        int renamed = 0, unchanged = 0;
        foreach (var root in roots)
        {
            foreach (Transform t in root.GetComponentsInChildren<Transform>(true))
            {
                if (!t.name.Contains("Row")) continue;

                string newName = Underscored(t.name);
                if (newName == t.name) { unchanged++; continue; }

                if (previewOnly)
                {
                    Debug.Log($"[RenameGearboxUnderscores] '{t.name}' -> '{newName}'");
                }
                else
                {
                    Undo.RecordObject(t.gameObject, "Rename");
                    t.gameObject.name = newName;
                    EditorUtility.SetDirty(t.gameObject);
                }
                renamed++;
            }
        }

        if (!previewOnly)
        {
            Undo.CollapseUndoOperations(group);
            if (renamed > 0) EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }
        Debug.Log($"[RenameGearboxUnderscores] {(previewOnly ? "Preview: " : "")}" +
                  $"{renamed} to rename, {unchanged} already fine. " +
                  (previewOnly ? "" : "Save the scene / prefab to persist."));
    }

    /// <summary>
    /// Insert underscores around the Row token and the side, if not already present.
    ///   BearingRow3Left -> Bearing_Row3_Left ;  GearRodRow1 -> GearRod_Row1
    /// Idempotent: a name that already has the underscores is returned unchanged.
    /// </summary>
    private static string Underscored(string name)
    {
        // underscore before "Row{N}" when directly preceded by a letter/digit
        string s = Regex.Replace(name, @"(?<=[A-Za-z0-9])(Row\d+)", "_$1");
        // underscore before the Left/Right side when directly following the row token
        s = Regex.Replace(s, @"(Row\d+)(Left|Right)", "$1_$2");
        return s;
    }
}
