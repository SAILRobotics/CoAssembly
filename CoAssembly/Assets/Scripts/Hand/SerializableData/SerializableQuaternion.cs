using UnityEngine;
using System;

using SerializableData;


namespace SerializableData 
{

    [Serializable]
    public struct SerializableQuaternion
    {
        public float x, y, z, w;
        public SerializableQuaternion(Quaternion q)
        {
            this.x = q.x;
            this.y = q.y;
            this.z = q.z;
            this.w = q.w;
        }
        public SerializableQuaternion(float x, float y, float z, float w)
        {
            this.x = x; 
            this.y = y; 
            this.z = z; 
            this.w = w;
        }

        public Quaternion ToUnityQuaternion() => new Quaternion(x, y, z, w);
    }

}