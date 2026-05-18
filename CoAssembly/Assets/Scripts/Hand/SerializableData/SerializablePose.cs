using UnityEngine;
using System;

using SerializableData;


namespace SerializableData 
{

    [Serializable]
    public class SerializablePose
    {
        public SerializableVector3 position;
        public SerializableQuaternion rotation;

        public SerializablePose() {}
        public SerializablePose(Vector3 pos, Quaternion rot)
        {
            position = new SerializableVector3(pos);
            rotation = new SerializableQuaternion(rot);
        }
        public SerializablePose(Pose pose)
        {
            position = new SerializableVector3(pose.position);
            rotation = new SerializableQuaternion(pose.rotation);
        }

        public Pose ToUnityPose()
        {
            return new Pose(
                new Vector3(position.x, position.y, position.z),
                new Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
            );
        }
    }

}
