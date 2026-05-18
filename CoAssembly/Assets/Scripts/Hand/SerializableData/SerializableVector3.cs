using UnityEngine;
using System;

using SerializableData;


namespace SerializableData 
{

    [Serializable]
    public struct SerializableVector3
    {
        public float x, y, z;
        public SerializableVector3(Vector3 v)
        {
            this. x = v.x;
            this. y = v.y;
            this. z = v.z;
        }
        public SerializableVector3(float x, float y, float z)
        {
            this.x = x; 
            this.y = y; 
            this.z = z;
        }

        public Vector3 ToUnityVector3() => new Vector3(x, y, z);
    }

}