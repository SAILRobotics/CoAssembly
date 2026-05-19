using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;

using Oculus.Interaction.Input;
using Newtonsoft.Json;
using SerializableData;

public class DualHandTrackingDataManager : MonoBehaviour
{
    [SerializeField] private OVRCameraRig _cameraRig;

    [Header("REAL / Tracked Hands (Oculus.Interaction.Input.Hand)")]
    [SerializeField] private Hand _leftHand;
    [SerializeField] private Hand _rightHand;

    [Header("SYNTH / Interaction Hands (Oculus.Interaction.Input.SyntheticHand)")]
    [SerializeField] private SyntheticHand _leftHandSynth;
    [SerializeField] private SyntheticHand _rightHandSynth;

    [Header("Gating (recommended)")]
    [SerializeField] private bool requireHighConfidenceForReal = true;
    [SerializeField] private bool requireConnectedTrackedForSynth = true;

    public HandTrackingSenderNetMQ _sender;
    public UnityEvent<string> JsonCompleteEvent;

    private float sendInterval = 1.0f / 120.0f; // (you wrote 20Hz but this is 120Hz)
    private float timeSinceLastSend = 0f;

    private int frameCount = 0;
    private float fpsTimer = 0f;
    private string lastJson = "";

    void Update()
    {
        var frame = CaptureFrame();
        string json = JsonConvert.SerializeObject(frame, Formatting.Indented);
        JsonCompleteEvent?.Invoke(json);
        lastJson = json;

        // Debug.Log($"HELLO: {lastJson}");

        timeSinceLastSend += Time.deltaTime;
        if (timeSinceLastSend >= sendInterval)
        {
            _sender?.SendJson(lastJson);
            timeSinceLastSend = 0f;
        }

        frameCount++;
        fpsTimer += Time.deltaTime;
        if (fpsTimer >= 1.0f)
        {
            frameCount = 0;
            fpsTimer = 0f;
        }
    }

    private SerializableTrackingData CaptureFrame()
    {
        var frame = new SerializableTrackingData();
        frame.timestamp = Time.time;

        // head
        if (_cameraRig != null && _cameraRig.centerEyeAnchor != null)
        {
            frame.head["CenterEye"] = new SerializablePose(
                _cameraRig.centerEyeAnchor.position,
                _cameraRig.centerEyeAnchor.rotation
            );
        }
        else
        {
            frame.head["CenterEye"] = null;
        }

        // REAL (keep original keys)
        frame.hands["LeftHand"]  = ShouldIncludeReal(_leftHand)  ? ExtractHandData(_leftHand)  : null;
        frame.hands["RightHand"] = ShouldIncludeReal(_rightHand) ? ExtractHandData(_rightHand) : null;

        // SYNTH (new keys)
        frame.hands["LeftHandSynth"]  = ShouldIncludeSynth(_leftHandSynth)  ? ExtractHandData(_leftHandSynth)  : null;
        frame.hands["RightHandSynth"] = ShouldIncludeSynth(_rightHandSynth) ? ExtractHandData(_rightHandSynth) : null;

        return frame;
    }

    private bool ShouldIncludeReal(Hand hand)
    {
        if (hand == null) return false;
        if (!requireHighConfidenceForReal) return true;
        return hand.IsHighConfidence;
    }

    private bool ShouldIncludeSynth(Hand hand) // SyntheticHand derives from Hand
    {
        if (hand == null) return false;
        if (!requireConnectedTrackedForSynth) return true;
        return hand.IsHighConfidence;  //&& hand.IsTracked;
    }

    private SerializableHandData ExtractHandData(Hand hand)
    {
        var handData = new SerializableHandData();

        foreach (var pair in TrackingUtils.HAND_JOINTS)
        {
            var name = pair.Key;
            var joints = pair.Value;

            var group = new List<SerializablePose>(joints.Length);
            for (int i = 0; i < joints.Length; i++)
            {
                Pose pose;
                if (hand.GetJointPose(joints[i], out pose))
                    group.Add(new SerializablePose(pose));
                else
                    group.Add(null);
            }

            handData.groups[name] = group;
        }

        handData.indexPinchStrength = hand.GetFingerPinchStrength(HandFinger.Index);
        return handData;
    }
}
