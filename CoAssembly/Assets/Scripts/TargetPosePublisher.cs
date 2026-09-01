using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;

/// <summary>
/// Streams the manipulated board pose back to Python (port 5013).
/// </summary>
public class TargetPosePublisher : MonoBehaviour
{
    [SerializeField] private int port = 5013;

    private static PublisherSocket _socket;
    private static int             _refCount;
    private static readonly object _lock = new();

    [Serializable]
    private class PoseMessage
    {
        public float[] tcp_pos;
        public float[] tcp_rot_xyzw;
        public string manipulation_state;
    }

    private void Start()
    {
        lock (_lock)
        {
            if (_socket == null)
            {
                AsyncIO.ForceDotNet.Force();
                _socket = new PublisherSocket();
                _socket.Bind($"tcp://0.0.0.0:{port}");
                NetMQManager.RegisterSender();
            }
            _refCount++;
        }
    }

    /// <summary>
    /// Send the board pose (in WorldRoot-local space) and manipulation state.
    /// Python back-calculates the corresponding TCP target from the board.
    /// </summary>
    public void SendPose(Vector3 localPos, Quaternion localRot,
                         string manipulationState = "released")
    {
        var msg = new PoseMessage
        {
            tcp_pos      = new[] { localPos.x, localPos.y, localPos.z },
            tcp_rot_xyzw = new[] { localRot.x, localRot.y, localRot.z, localRot.w },
            manipulation_state = manipulationState,
        };
        lock (_lock)
        {
            try { _socket?.SendFrame(JsonConvert.SerializeObject(msg)); }
            catch (Exception e) { Debug.LogWarning("[TargetPosePublisher] " + e.Message); }
        }
        if (manipulationState != "dragging")
            Debug.Log($"[TargetPosePublisher] Sent {manipulationState} pose {localPos}");
    }

    private void OnDestroy()
    {
        lock (_lock)
        {
            _refCount = Mathf.Max(0, _refCount - 1);
            if (_refCount == 0 && _socket != null)
            {
                _socket.Close(); _socket.Dispose(); _socket = null;
                NetMQManager.UnregisterSender();
            }
        }
    }
}
