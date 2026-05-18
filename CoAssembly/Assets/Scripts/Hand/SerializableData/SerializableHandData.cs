using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

using SerializableData;


namespace SerializableData 
{

    [Serializable]
    public class SerializableHandData 
    {
        public Dictionary<string, List<SerializablePose>> groups = new();
        public float indexPinchStrength = 0f;
    }

}