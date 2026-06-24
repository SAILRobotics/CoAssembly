using System;
using System.Collections.Generic;

[Serializable]
public class JointValueMessage
{
    public float timestamp;
    public List<float> joint_values;
}