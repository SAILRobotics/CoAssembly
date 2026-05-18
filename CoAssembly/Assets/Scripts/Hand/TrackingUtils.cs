using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

using Oculus.Interaction;
using Oculus.Interaction.Input;


public static class TrackingUtils
{
    public static readonly Dictionary<string, HandJointId[]> HAND_JOINTS = new Dictionary<string, HandJointId[]> {
        { "Wrist", new[] {HandJointId.HandWristRoot} },  
        { "Palm", new[] {HandJointId.HandPalm} }, 
        { "Thumb", new[] {HandJointId.HandThumb1, HandJointId.HandThumb2, HandJointId.HandThumb3, HandJointId.HandThumbTip} }, 
        { "Index", new[] {HandJointId.HandIndex1, HandJointId.HandIndex2, HandJointId.HandIndex3, HandJointId.HandIndexTip} }, 
        { "Middle", new[] {HandJointId.HandMiddle1, HandJointId.HandMiddle2, HandJointId.HandMiddle3, HandJointId.HandMiddleTip} }, 
        { "Ring", new[] {HandJointId.HandRing1, HandJointId.HandRing2, HandJointId.HandRing3, HandJointId.HandRingTip} }, 
        { "Pinky", new[] {HandJointId.HandPinky1, HandJointId.HandPinky2, HandJointId.HandPinky3, HandJointId.HandPinkyTip} }  
    };
}
