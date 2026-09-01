using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Restores the complete, configured AR Handle hierarchy from the known-good
/// WorkholdingTesting scene into WorkHoldingTestNew. The operation is
/// idempotent: running it again repairs references rather than making copies.
/// </summary>
public static class RestoreWorkholdingArHandle
{
    private const string SourceScene =
        "Assets/Scenes/WorkholdingTesting.unity";
    private const string TargetScene =
        "Assets/Scenes/WorkHoldingTestNew.unity";

    [MenuItem("CoAssembly/Study 2/Restore AR Handle in WorkHoldingTestNew")]
    public static void Restore()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            return;

        Scene target = EditorSceneManager.OpenScene(TargetScene, OpenSceneMode.Single);
        GripStateReceiver receiver = FindInScene<GripStateReceiver>(target);
        if (receiver == null)
            throw new InvalidOperationException("WorkHoldingTestNew has no GripStateReceiver.");

        GameObject handle = FindGameObject(target, "AR Handle");
        if (handle == null)
        {
            Scene source = EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Additive);
            GameObject sourceHandle = FindGameObject(source, "AR Handle");
            if (sourceHandle == null)
                throw new InvalidOperationException("Source scene has no AR Handle.");

            SceneManager.SetActiveScene(target);
            handle = UnityEngine.Object.Instantiate(sourceHandle);
            handle.name = "AR Handle";
            SceneManager.MoveGameObjectToScene(handle, target);
            EditorSceneManager.CloseScene(source, true);
        }

        Transform parent = receiver.worldRoot != null
            ? receiver.worldRoot : FindGameObject(target, "WorldRoot")?.transform;
        if (parent == null)
            throw new InvalidOperationException("WorkHoldingTestNew has no WorldRoot.");
        handle.transform.SetParent(parent, false);

        ARManipulationHandle manipulation =
            handle.GetComponent<ARManipulationHandle>();
        if (manipulation == null)
            throw new InvalidOperationException("Restored handle lacks ARManipulationHandle.");

        TargetPosePublisher publisher = FindInScene<TargetPosePublisher>(target);
        manipulation.arBox = receiver.arBox;
        manipulation.arHandle = handle;
        manipulation.targetPublisher = publisher;
        manipulation.worldRoot = parent;
        receiver.arHandle = handle;
        receiver.manipulationHandle = manipulation;
        handle.SetActive(false);

        EditorUtility.SetDirty(manipulation);
        EditorUtility.SetDirty(receiver);
        EditorSceneManager.MarkSceneDirty(target);
        EditorSceneManager.SaveScene(target);
        Selection.activeGameObject = handle;
        Debug.Log("[Study2] Restored and wired AR Handle in WorkHoldingTestNew.");
    }

    private static T FindInScene<T>(Scene scene) where T : Component
    {
        return scene.GetRootGameObjects()
            .SelectMany(root => root.GetComponentsInChildren<T>(true))
            .FirstOrDefault();
    }

    private static GameObject FindGameObject(Scene scene, string objectName)
    {
        Transform transform = scene.GetRootGameObjects()
            .SelectMany(root => root.GetComponentsInChildren<Transform>(true))
            .FirstOrDefault(candidate => candidate.name == objectName);
        return transform != null ? transform.gameObject : null;
    }
}
