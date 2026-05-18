using UnityEngine;
using UnityEngine.Events;
using System;
using System.Collections;
using System.Collections.Generic;

using Oculus.Interaction;
using Oculus.Interaction.Input;

using Newtonsoft.Json;
using SerializableData;

public class TrackingDataManager : MonoBehaviour
{
    [SerializeField]
    private OVRCameraRig _cameraRig;

    [SerializeField]
    private Hand _leftHand;

    [SerializeField]
    private Hand _rightHand;

    public HandTrackingSenderNetMQ _sender;
    public UnityEvent<string> JsonCompleteEvent;

    private float sendInterval = 1.0f / 120.0f;  // 20Hz
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

        // 20Hz send
        timeSinceLastSend += Time.deltaTime;
        if (timeSinceLastSend >= sendInterval)
        {
            if (_sender != null)
                _sender.SendJson(lastJson);
            timeSinceLastSend = 0f;
        }

        // FPS counter
        frameCount++;
        fpsTimer += Time.deltaTime;
        if (fpsTimer >= 1.0f)
        {
            // Debug.Log($"📸 FPS: {frameCount}");
            frameCount = 0;
            fpsTimer = 0f;
        }
    }

    private SerializableTrackingData CaptureFrame()
    {
        var frame = new SerializableTrackingData();

        // timestamp
        frame.timestamp = Time.time;

        // head
        if (_cameraRig != null && _cameraRig.centerEyeAnchor != null)
        {
            var centerEyeData = new SerializablePose(
                _cameraRig.centerEyeAnchor.position,
                _cameraRig.centerEyeAnchor.rotation
            );
            frame.head.Add("CenterEye", centerEyeData);
        }
        else
        {
            frame.head.Add("CenterEye", null);
        }

        // left hand
        if (_leftHand != null && _leftHand.IsHighConfidence)
        {
            var leftHandData = ExtractHandData(_leftHand);
            frame.hands.Add("LeftHand", leftHandData);
        }
        else
        {
            frame.hands.Add("LeftHand", null);
        }

        // right hand
        if (_rightHand != null && _rightHand.IsHighConfidence)
        {
            var rightHandData = ExtractHandData(_rightHand);
            frame.hands.Add("RightHand", rightHandData);
        }
        else
        {
            frame.hands.Add("RightHand", null);
        }

        return frame;
    }

    private SerializableHandData ExtractHandData(Hand hand)
    {
        var handData = new SerializableHandData();

        foreach (var pair in TrackingUtils.HAND_JOINTS)
        {
            var name = pair.Key;
            var joints = pair.Value;

            List<SerializablePose> group = new List<SerializablePose>();
            for (int i = 0; i < joints.Length; i++)
            {
                Pose pose = Pose.identity;
                if (hand.GetJointPose(joints[i], out pose))
                {
                    group.Add(new SerializablePose(pose));
                }
                else
                {
                    group.Add(null);
                }
            }

            handData.groups.Add(name, group);

        }
        handData.indexPinchStrength = hand.GetFingerPinchStrength(HandFinger.Index);

        return handData;
    }
}
