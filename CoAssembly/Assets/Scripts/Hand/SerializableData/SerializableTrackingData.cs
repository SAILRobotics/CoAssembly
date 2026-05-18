using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

using SerializableData;


namespace SerializableData 
{

    [Serializable]
    public class SerializableTrackingData 
    {
        public float timestamp;
        public Dictionary<string, SerializablePose> head = new();
        public Dictionary<string, SerializableHandData> hands = new();
    }
    
}