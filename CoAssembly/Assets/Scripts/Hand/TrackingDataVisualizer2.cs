using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

using Oculus.Interaction;
using Oculus.Interaction.Input;

using Newtonsoft.Json;

using SerializableData;


public class TrackingDataVisualizer2 : MonoBehaviour
{
    [SerializeField]
    private GameObject _jointPrefab;

    [SerializeField]
    private GameObject _frustumPrefab;

    private GameObject _headWrapper;
    private GameObject _leftHandWrapper;
    private GameObject _rightHandWrapper;
    
    private GameObject _centerEyeFrustum;
    private Dictionary<string, GameObject> _leftHandJoints;
    private Dictionary<string, GameObject> _rightHandJoints;


    void Start()
    {
        PrebuildMarkers();
    }


    void Update()
    {
        
    }

    private void PrebuildMarkers() 
    {
        _headWrapper = new GameObject("Head");
        _headWrapper.transform.SetParent(this.transform);
        _headWrapper.SetActive(false);
        _centerEyeFrustum = Instantiate(_frustumPrefab, _headWrapper.transform);
        _centerEyeFrustum.name = "CenterEye";
        _centerEyeFrustum.SetActive(false);

        _leftHandWrapper = new GameObject("LeftHand");
        _leftHandWrapper.transform.SetParent(this.transform);
        _leftHandWrapper.SetActive(false);
        _leftHandJoints = new Dictionary<string, GameObject>();
        foreach (var pair in TrackingUtils.HAND_JOINTS)
        {
            string name = pair.Key;
            var joints = pair.Value;
            for (int i = 0; i < joints.Length; i++)
            {
                string jointName = $"Left_{name}_{i}";
                GameObject marker = Instantiate(_jointPrefab, _leftHandWrapper.transform);
                marker.name = jointName;
                marker.SetActive(false);
                _leftHandJoints.Add(jointName, marker);
            }
        }

        _rightHandWrapper = new GameObject("RightHand");
        _rightHandWrapper.transform.SetParent(this.transform);
        _rightHandWrapper.SetActive(false);
        _rightHandJoints = new Dictionary<string, GameObject>();
        foreach (var pair in TrackingUtils.HAND_JOINTS)
        {
            string name = pair.Key;
            var joints = pair.Value;
            for (int i = 0; i < joints.Length; i++)
            {
                string jointName = $"Right_{name}_{i}";
                GameObject marker = Instantiate(_jointPrefab, _rightHandWrapper.transform);
                marker.name = jointName;
                marker.SetActive(false);
                _rightHandJoints.Add(jointName, marker);
            }
        }
    }

    public void UpdateMarkers(string json)  
    {   
        SerializableTrackingData trackingData = JsonConvert.DeserializeObject<SerializableTrackingData>(json);
    
        // head
        var centerEyePoseData = trackingData.head["CenterEye"];
        if (centerEyePoseData == null) 
        {
            _headWrapper.SetActive(false);
            _centerEyeFrustum.SetActive(false);
        }
        else 
        {
            _headWrapper.SetActive(true);
            var centerEyePose = centerEyePoseData.ToUnityPose();
            _centerEyeFrustum.transform.position = centerEyePose.position;
            _centerEyeFrustum.transform.rotation = centerEyePose.rotation;
            _centerEyeFrustum.SetActive(true);
        }

        // hands
        foreach (var hand in trackingData.hands) 
        {
            string handName = hand.Key;
            var handData = hand.Value;

            string handedness;
            GameObject handWrapper;
            Dictionary<string, GameObject> handJoints;
            if (handName == "LeftHand") 
            {
                handedness = "Left";
                handWrapper = _leftHandWrapper;
                handJoints = _leftHandJoints;
            }
            else if (handName == "RightHand") 
            {
                handedness = "Right";
                handWrapper = _rightHandWrapper;
                handJoints = _rightHandJoints;
            }
            else {
                handedness = "None";
                handWrapper = null;
                handJoints = null;
            }    
            
            if (handData == null) 
            {
                handWrapper.SetActive(false);
                continue;
            }
            if (handedness == "None") 
            {
                continue;
            }
            handWrapper.SetActive(true);

            foreach (var groupPair in handData.groups) 
            {
                string groupName = groupPair.Key;
                var joints = groupPair.Value;

                for (int i = 0; i < joints.Count; i++)
                {
                    var poseData = joints[i];

                    string jointName = $"{handedness}_{groupName}_{i}";
                    var marker = handJoints[jointName];

                    if (poseData == null) 
                    {
                        marker.transform.position = Vector3.zero;
                        marker.transform.rotation = Quaternion.identity;
                        marker.SetActive(false);
                        continue;
                    }
                    else 
                    {
                        var pose = poseData.ToUnityPose();
                        marker.transform.position = pose.position;
                        marker.transform.rotation = pose.rotation;
                        marker.SetActive(true);
                    }

                }
            }
        }
    }
}
