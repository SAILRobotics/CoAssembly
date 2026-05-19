using UnityEngine;
using System.Collections.Generic;

using Newtonsoft.Json;
using SerializableData;


public class DualHandVisualizer : MonoBehaviour
{
    [Header("Prefabs")]
    [SerializeField] private GameObject _jointPrefab;        // REAL
    [SerializeField] private GameObject _synthJointPrefab;   // SYNTH (different visuals)
    [SerializeField] private GameObject _frustumPrefab;

    [Header("Debug")]
    [SerializeField] private bool log = false;

    [Header("Visibility (keep colliders working)")]
    [Tooltip("If ON: hides joint visuals (Renderer.enabled=false) but keeps GameObjects active so colliders still work.")]
    [SerializeField] private bool _hideVisualsButKeepColliders = false;

    private GameObject _headWrapper;
    private GameObject _centerEyeFrustum;

    // REAL wrappers + joints
    private GameObject _leftHandWrapper;
    private GameObject _rightHandWrapper;
    private Dictionary<string, GameObject> _leftHandJoints;
    private Dictionary<string, GameObject> _rightHandJoints;

    // SYNTH wrappers + joints
    private GameObject _leftHandSynthWrapper;
    private GameObject _rightHandSynthWrapper;
    private Dictionary<string, GameObject> _leftHandSynthJoints;
    private Dictionary<string, GameObject> _rightHandSynthJoints;

    void Start()
    {
        PrebuildMarkers();
        ApplyHideVisualsSettingAll(); // apply initial toggle state
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        if (!Application.isPlaying) return;
        ApplyHideVisualsSettingAll(); // lets you toggle in inspector while playing
    }
#endif

    private void PrebuildMarkers()
    {
        // Head
        _headWrapper = new GameObject("Head");
        _headWrapper.transform.SetParent(this.transform, false);
        _headWrapper.SetActive(false);

        _centerEyeFrustum = Instantiate(_frustumPrefab, _headWrapper.transform);
        _centerEyeFrustum.name = "CenterEye";
        _centerEyeFrustum.SetActive(false);

        // REAL: Left
        _leftHandWrapper = new GameObject("LeftHand");
        _leftHandWrapper.transform.SetParent(this.transform, false);
        _leftHandWrapper.SetActive(true); // keep active so colliders can work
        _leftHandJoints = BuildJointMarkers(_leftHandWrapper.transform, "Left", _jointPrefab);

        // REAL: Right
        _rightHandWrapper = new GameObject("RightHand");
        _rightHandWrapper.transform.SetParent(this.transform, false);
        _rightHandWrapper.SetActive(true);
        _rightHandJoints = BuildJointMarkers(_rightHandWrapper.transform, "Right", _jointPrefab);

        // SYNTH: Left
        _leftHandSynthWrapper = new GameObject("LeftHandSynth");
        _leftHandSynthWrapper.transform.SetParent(this.transform, false);
        _leftHandSynthWrapper.SetActive(true);
        _leftHandSynthJoints = BuildJointMarkers(_leftHandSynthWrapper.transform, "LeftSynth", _synthJointPrefab);

        // SYNTH: Right
        _rightHandSynthWrapper = new GameObject("RightHandSynth");
        _rightHandSynthWrapper.transform.SetParent(this.transform, false);
        _rightHandSynthWrapper.SetActive(true);
        _rightHandSynthJoints = BuildJointMarkers(_rightHandSynthWrapper.transform, "RightSynth", _synthJointPrefab);

        if (log)
        {
            Debug.Log($"[DualHandVisualizer] Built joints: " +
                      $"L={_leftHandJoints.Count}, R={_rightHandJoints.Count}, " +
                      $"LSynth={_leftHandSynthJoints.Count}, RSynth={_rightHandSynthJoints.Count}", this);
        }
    }

    private Dictionary<string, GameObject> BuildJointMarkers(Transform parent, string prefix, GameObject prefab)
    {
        var dict = new Dictionary<string, GameObject>();

        if (prefab == null)
        {
            if (log) Debug.LogWarning($"[DualHandVisualizer] Prefab NULL for {prefix}. No markers created.", this);
            return dict;
        }

        foreach (var pair in TrackingUtils.HAND_JOINTS)
        {
            string groupName = pair.Key;
            var joints = pair.Value;

            for (int i = 0; i < joints.Length; i++)
            {
                string jointName = $"{prefix}_{groupName}_{i}";
                GameObject marker = Instantiate(prefab, parent);
                marker.name = jointName;

                // Start disabled visually (matches your prior behavior), but keep GO active.
                marker.SetActive(true);
                SetMarkerVisible(marker, false);

                dict.Add(jointName, marker);
            }
        }

        return dict;
    }

    // ---------------------------
    // Hide visuals but keep colliders
    // ---------------------------
    private void ApplyHideVisualsSettingAll()
    {
        ApplyHideVisualsSetting(_leftHandJoints);
        ApplyHideVisualsSetting(_rightHandJoints);
        ApplyHideVisualsSetting(_leftHandSynthJoints);
        ApplyHideVisualsSetting(_rightHandSynthJoints);
    }

    private void ApplyHideVisualsSetting(Dictionary<string, GameObject> dict)
    {
        if (dict == null) return;
        foreach (var kv in dict)
        {
            if (kv.Value == null) continue;

            // If hiding visuals globally, force invisible.
            // If not hiding globally, don't force visible here; runtime will control visibility per joint.
            if (_hideVisualsButKeepColliders)
                SetMarkerVisible(kv.Value, false);
        }
    }

    private void SetMarkerVisible(GameObject marker, bool visible)
    {
        if (marker == null) return;
        var renderers = marker.GetComponentsInChildren<Renderer>(true);
        for (int i = 0; i < renderers.Length; i++)
        {
            if (renderers[i] != null) renderers[i].enabled = visible;
        }
    }

    private void SetAllMarkersVisible(Dictionary<string, GameObject> dict, bool visible)
    {
        if (dict == null) return;
        foreach (var kv in dict)
        {
            if (kv.Value != null) SetMarkerVisible(kv.Value, visible);
        }
    }

    // Hook this to JsonCompleteEvent
    public void UpdateMarkers(string json)
    {
        if (string.IsNullOrEmpty(json))
        {
            if (log) Debug.LogWarning("[DualHandVisualizer] json is null/empty", this);
            return;
        }

        SerializableTrackingData trackingData;
        try
        {
            trackingData = JsonConvert.DeserializeObject<SerializableTrackingData>(json);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[DualHandVisualizer] JSON parse error: {e.Message}", this);
            return;
        }

        if (trackingData == null) return;

        // --- head ---
        SerializablePose centerEyePoseData = null;
        if (trackingData.head != null && trackingData.head.TryGetValue("CenterEye", out centerEyePoseData) && centerEyePoseData != null)
        {
            _headWrapper.SetActive(true);
            var centerEyePose = centerEyePoseData.ToUnityPose();
            _centerEyeFrustum.transform.SetPositionAndRotation(centerEyePose.position, centerEyePose.rotation);
            _centerEyeFrustum.SetActive(true);
        }
        else
        {
            _headWrapper.SetActive(false);
            _centerEyeFrustum.SetActive(false);
        }

        // --- hands ---
        if (trackingData.hands == null) return;

        foreach (var hand in trackingData.hands)
        {
            string handName = hand.Key;
            var handData = hand.Value;

            GameObject handWrapper = null;
            Dictionary<string, GameObject> handJoints = null;
            string prefix = null;

            // REAL
            if (handName == "LeftHand")
            {
                handWrapper = _leftHandWrapper;
                handJoints = _leftHandJoints;
                prefix = "Left";
            }
            else if (handName == "RightHand")
            {
                handWrapper = _rightHandWrapper;
                handJoints = _rightHandJoints;
                prefix = "Right";
            }
            // SYNTH
            else if (handName == "LeftHandSynth")
            {
                handWrapper = _leftHandSynthWrapper;
                handJoints = _leftHandSynthJoints;
                prefix = "LeftSynth";
            }
            else if (handName == "RightHandSynth")
            {
                handWrapper = _rightHandSynthWrapper;
                handJoints = _rightHandSynthJoints;
                prefix = "RightSynth";
            }
            else
            {
                continue;
            }

            if (handWrapper == null) continue;

            if (handData == null)
            {
                // Keep wrapper active (colliders), but hide visuals for that hand
                if (handJoints != null)
                    SetAllMarkersVisible(handJoints, false);
                continue;
            }

            // Always keep wrapper active so colliders are alive
            handWrapper.SetActive(true);

            if (handData.groups == null) continue;

            foreach (var groupPair in handData.groups)
            {
                string groupName = groupPair.Key;
                var joints = groupPair.Value;
                if (joints == null) continue;

                for (int i = 0; i < joints.Count; i++)
                {
                    var poseData = joints[i];
                    string jointName = $"{prefix}_{groupName}_{i}";

                    if (handJoints == null || !handJoints.TryGetValue(jointName, out var marker) || marker == null)
                        continue;

                    // Keep marker GameObject ACTIVE always (colliders work)
                    if (!marker.activeSelf) marker.SetActive(true);

                    if (poseData == null)
                    {
                        // If pose missing -> hide visuals only
                        SetMarkerVisible(marker, false);
                        continue;
                    }

                    var pose = poseData.ToUnityPose();
                    marker.transform.SetPositionAndRotation(pose.position, pose.rotation);

                    // Visual policy:
                    // - If global hide is ON -> keep invisible
                    // - Else -> visible
                    if (_hideVisualsButKeepColliders)
                        SetMarkerVisible(marker, false);
                    else
                        SetMarkerVisible(marker, true);
                }
            }
        }
    }
}
