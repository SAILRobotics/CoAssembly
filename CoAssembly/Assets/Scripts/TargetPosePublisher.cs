using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json;
using System;

/// <summary>
/// Sends the manipulated TCP target pose back to Python (port 5013).
/// Call SendPose() from ARManipulationHandle when the user releases the handle.
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
    /// Send the final TCP pose (in WorldRoot-local space) to Python.
    /// Pass the box transform — Python will back-calculate TCP from the box.
    /// </summary>
    public void SendPose(Vector3 localPos, Quaternion localRot)
    {
        var msg = new PoseMessage
        {
            tcp_pos      = new[] { localPos.x, localPos.y, localPos.z },
            tcp_rot_xyzw = new[] { localRot.x, localRot.y, localRot.z, localRot.w },
        };
        lock (_lock)
        {
            try { _socket?.SendFrame(JsonConvert.SerializeObject(msg)); }
            catch (Exception e) { Debug.LogWarning("[TargetPosePublisher] " + e.Message); }
        }
        Debug.Log($"[TargetPosePublisher] Sent target pose {localPos}");
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
