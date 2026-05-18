import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.Set = collections.abc.Set
collections.Iterable = collections.abc.Iterable

import fractions
import math
fractions.gcd = math.gcd

import os
import sys
import time
import zmq
from typing import Dict, List, Optional, Any
import json
from scipy.spatial.transform import Rotation as R

import math
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

if not hasattr(np, "int_"):
    np.int_ = np.int64

# Optional: also restore the non-underscore aliases, in case some lib uses them
if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "int"):
    np.int = int

from robot_controller.robot_controller import RobotController
from main_scene.visualizer import Visualizer
from main_scene.zmq_config import ZMQConfig
from calib_and_object.object_pose_estimator import ObjectModel
from utils.unity_conversion import (
    unity_to_open3d_vector,
    unity_to_open3d_vector2, 
    unity_to_open3d_quaternion,
    open3d_to_unity_vector,
    open3d_to_unity_quaternion,
    open3d_to_unity_vector_array,
    rotation_matrix_to_quaternion,
    transform_point_world_to_robot,
    get_2d_bbox,
    find_the_closest_obj,
    get_optimal_robot_pose_for_object,
    HAND_BONES
)
import rtde_receive

class MainScene:
    def __init__(
        self,
        object_pose_estimator=None,
        sensor_receiver=None,
        real_object: bool = True, 
        transfer_to_ubuntu: bool = True,
        connect_robot: bool = False,
        visualize: bool = True
    ):  
        
        self.object_pose_estimator = object_pose_estimator
        self.sensor_receiver = sensor_receiver

        self.real_object = real_object

       
        self.verbose = True
        self.transfer_to_ubuntu = transfer_to_ubuntu
        self.connect_robot = connect_robot
        
        self.robot_ip = "192.168.50.70"
        self.visualize = visualize

        if (self.visualize):
            self.visualizer = Visualizer(object_pose_estimator=object_pose_estimator)

        if (self.connect_robot):
            self.init_robot()
        #-----Communication Related------
        self.robot_status = None                            #robot status
        self.gripper_status = None                          #gripper status 
        self.robot_joint_angles_radians = None              #robot joint angles in radians

        self.T_world_cam_dict = {}
        self.T_robot_in_world = None                        #from object_pose_estimator
        self.real_hand_data = None                              #receive hand data from unity
        self.synthetic_hand_data = None                              #receive synthetic hand data from unity

        self.objects: List[ObjectModel] = self.object_pose_estimator.objects
        self.objects_in_scene = []
        #---------------------------Communication Channels------------------------------
        self.zmq_config = ZMQConfig()
        self.zmq_context = None

        # Linux (Ubuntu) <-> Windows
        self.robot_status_sub_from_linux = None     # receive robot status from Ubuntu
        self.robot_pose_pub_to_linux = None         # send calibrated robot pose to Ubuntu
        self.hand_pub_to_linux = None               # send hand data to Ubuntu (optional, TODO)

        # Windows <-> Unity
        self.object_pose_pub_to_unity = None        # send object poses to Unity
        self.object_pose_pub_to_linux = None        # send object poses to Linux
        self.object_pose_sub_from_unity = None      # receive updated object poses from Unity

        self.initial_robot_pose_pub_to_unity = None # send calibrated robot pose to Unity
        self.initial_camera_pose_pub_to_unity = None# send initial camera pose to Unity
        self.workspace_bounds_pub_to_unity = None   # send workspace bounds to Unity

        self.robot_joint_pub_to_unity = None        # send robot joint angles to Unity
        self.gripper_pub_to_unity = None            # send gripper info to Unity

        self.hand_sub1_from_unity = None            # receive hand data from Unity
        self.hand_sub2_from_unity = None            # receive synthetic hand data from Unity

        #---------------------------- Timing / rate limiting ---------------------------
        now = time.time()
        self.start_time = now
       
        # Frequencies in Hz (set to 0 or None = no throttling)
        self.tx_rate_hz = {
            "tx_object_poses_unity": 30.0,
            "tx_object_poses_linux": 30.0,
            "tx_gripper_state_unity": 60.0,
            "tx_robot_joints_unity": 60.0,
            "tx_initial_robot_pose_unity": 1.0,    # usually low / or one-shot
            "tx_initial_camera_pose_unity": 1.0,   # usually low / or one-shot
            "tx_robot_pose_to_ubuntu": 1.0,        # usually low / or one-shot
            "tx_workspace_bounds_unity": 1.0,
            "tx_hand_to_ubuntu": 30.0,             # if you use hand_pub_to_linux
        }

        self.rx_rate_hz = {
            "rx_robot_status_from_ubuntu": 0.0,    # 0 => no throttle (whenever available)
            "rx_hand_data1_from_unity": 30.0,
            # "rx_hand_data2_from_unity": 120.0,
            "rx_object_pose_from_unity": 30.0,
        }

        # last timestamps (seconds)
        self.last_tx_time = {k: now for k in self.tx_rate_hz.keys()}
        self.last_rx_time = {k: now for k in self.rx_rate_hz.keys()}

        # One-shot flags (if you want)
        self.sent_initial_robot_pose_to_unity = False
        self.sent_initial_robot_pose_to_linux = False
        self.sent_initial_camera_pose_to_unity = False
        
        self.setup_communication()

        for name in self.object_pose_estimator.camera_names:
            cam_matrix = self.object_pose_estimator.cam_matrices[name]
            T_world_cam = self.object_pose_estimator.extrinsics[name]
            self.T_world_cam_dict[name] = T_world_cam

        if "T_robot_in_world" in self.object_pose_estimator.extrinsics:
            self.T_robot_in_world = self.object_pose_estimator.extrinsics["T_robot_in_world"]
            
    def init_robot(self):
        """
        Initialize RobotController on Ubuntu and connect to UR.
        """
        if self.verbose:
            print(f"[MainSceneWindows] Initializing RobotController for {self.robot_ip}...")

        self.robot = RobotController(self.robot_ip, control_enabled=False, verbose=True, use_gripper=False)
        self.robot.connect()


    def build_robot_status_msg(self) -> Dict[str, Any]:
        """
        Read robot state & gripper, pack into a dict for ZMQ send_json.
        """
        if self.robot is None or not self.connect_robot:
            return {
                "timestamp": time.time(),
                "connected": False,
                "joints": None,
                "tcp": None,
                "gripper": None,
                "safety_raw": None,
                "safety_flags": ["NO_ROBOT"],
                "safe_to_move": False,
            }

        # Joints and TCP
        try:
            q = self.robot.get_q()     # List[float]
        except Exception as e:
            if self.verbose:
                print(f"[MainSceneLinux] WARNING: get_q failed: {e}")
            q = None

        try:
            tcp = self.robot.get_robot_tcp()
        except Exception as e:
            if self.verbose:
                print(f"[MainSceneLinux] WARNING: get_robot_tcp failed: {e}")
            tcp = None

        # Safety
        s_raw = self.robot.get_safety_status()
        s_flags = self.robot.decode_safety(s_raw)
        safe = self.robot.is_safe_to_move()

        # Gripper
        grip_state = self.robot.get_gripper_state()

        return {
            "timestamp": time.time(),
            "connected": self.robot.is_connected(),
            "joints": q,
            "tcp": tcp,
            "gripper": grip_state,
            "safety_raw": s_raw,
            "safety_flags": s_flags,
            "safe_to_move": safe,
        }
    
    # ---------- COMMUNICATION PLACEHOLDER ----------
    def setup_communication(self):
        cfg = self.zmq_config
        ctx = zmq.Context.instance()
        self.zmq_context = ctx

        # -------------------------
        # (1) Robot Status: Ubuntu -> Windows (Ubuntu PUB, Windows SUB)
        # Ubuntu binds, Windows connects
        # -------------------------
        self.robot_status_sub_from_linux = ctx.socket(zmq.SUB)
        self.robot_status_sub_from_linux.setsockopt(zmq.CONFLATE, 1)
        self.robot_status_sub_from_linux.setsockopt_string(zmq.SUBSCRIBE, "")
        self.robot_status_sub_from_linux.connect(cfg.to_ubuntu(cfg.robot_status_port_from_ubuntu))
        if self.verbose:
            print(f"[MainScene] RobotStatus SUB (from Ubuntu) connected to {cfg.to_ubuntu(cfg.robot_status_port_from_ubuntu)}")

        # -------------------------
        # (2) Calibrated robot pose: Windows -> Ubuntu (Windows PUB, Ubuntu SUB)
        # Windows binds, Ubuntu connects
        # -------------------------
        self.robot_pose_pub_to_linux = ctx.socket(zmq.PUB)
        self.robot_pose_pub_to_linux.setsockopt(zmq.SNDHWM, 1)
        self.robot_pose_pub_to_linux.setsockopt(zmq.LINGER, 0)
        self.robot_pose_pub_to_linux.bind(cfg.to_win(cfg.robot_pose_port_to_ubuntu))
        if self.verbose:
            print(f"[MainScene] RobotPose PUB (to Ubuntu) bound on {cfg.to_win(cfg.robot_pose_port_to_ubuntu)}")
        
            

        # -------------------------
        # (3) Object poses: Windows -> Unity (Windows PUB, Unity SUB)
        # (4) Object poses: Windows -> Ubuntu (Windows PUB, Ubuntu SUB)
        # (4b) Updated object poses: Unity -> Windows (Unity PUB, Windows SUB)
        # -------------------------
        
        # PUB to Unity
        self.object_pose_pub_to_unity = ctx.socket(zmq.PUB)
        self.object_pose_pub_to_unity.setsockopt(zmq.SNDHWM, 1)
        self.object_pose_pub_to_unity.setsockopt(zmq.LINGER, 0)
        self.object_pose_pub_to_unity.bind(cfg.to_unity(cfg.object_pose_port_to_unity))
        if self.verbose:
            print(f"[MainScene] ObjectPose PUB (to Unity) bound on {cfg.to_unity(cfg.object_pose_port_to_unity)}")

        # PUB to Ubuntu
        self.object_pose_pub_to_linux = ctx.socket(zmq.PUB)
        self.object_pose_pub_to_linux.setsockopt(zmq.SNDHWM, 1)
        self.object_pose_pub_to_linux.setsockopt(zmq.LINGER, 0)
        self.object_pose_pub_to_linux.bind(cfg.to_win(cfg.object_pose_port_to_ubuntu))
        if self.verbose:
            print(f"[MainScene] ObjectPose PUB (to Ubuntu) bound on {cfg.to_win(cfg.object_pose_port_to_ubuntu)}")

        # SUB from Unity (object updates)
        self.object_pose_sub_from_unity = ctx.socket(zmq.SUB)
        self.object_pose_sub_from_unity.setsockopt(zmq.CONFLATE, 1)
        self.object_pose_sub_from_unity.setsockopt_string(zmq.SUBSCRIBE, "")
        self.object_pose_sub_from_unity.connect(cfg.to_unity(cfg.obj_pose_port_from_unity))
        if self.verbose:
            print(f"[MainScene] ObjectPose SUB (from Unity) connected to {cfg.to_unity(cfg.obj_pose_port_from_unity)}")

        # -------------------------
        # (5) Gripper state: Windows -> Unity (Windows PUB, Unity SUB)
        # Windows binds, Unity connects
        # -------------------------
        self.gripper_pub_to_unity = ctx.socket(zmq.PUB)
        self.gripper_pub_to_unity.setsockopt(zmq.SNDHWM, 1)
        self.gripper_pub_to_unity.setsockopt(zmq.LINGER, 0)
        self.gripper_pub_to_unity.bind(cfg.to_unity(cfg.gripper_port_to_unity))
        if self.verbose:
            print(f"[MainScene] Gripper PUB (to Unity) bound on {cfg.to_unity(cfg.gripper_port_to_unity)}")

        # -------------------------
        # (6) Robot joints: Windows -> Unity (Windows PUB, Unity SUB)
        # Windows binds, Unity connects
        # -------------------------
        self.robot_joint_pub_to_unity = ctx.socket(zmq.PUB)
        self.robot_joint_pub_to_unity.setsockopt(zmq.SNDHWM, 1)
        self.robot_joint_pub_to_unity.setsockopt(zmq.LINGER, 0)
        self.robot_joint_pub_to_unity.bind(cfg.to_unity(cfg.robot_joint_port_to_unity))
        if self.verbose:
            print(f"[MainScene] RobotJoints PUB (to Unity) bound on {cfg.to_unity(cfg.robot_joint_port_to_unity)}")

        # -------------------------
        # (7) Initial robot pose: Windows -> Unity (Windows PUB, Unity SUB)
        # Windows binds, Unity connects
        # -------------------------
        self.initial_robot_pose_pub_to_unity = ctx.socket(zmq.PUB)
        self.initial_robot_pose_pub_to_unity.setsockopt(zmq.SNDHWM, 1)
        self.initial_robot_pose_pub_to_unity.setsockopt(zmq.LINGER, 0)
        self.initial_robot_pose_pub_to_unity.bind(cfg.to_unity(cfg.robot_pose_port_to_unity))
        if self.verbose:
            print(f"[MainScene] InitialRobotPose PUB (to Unity) bound on {cfg.to_unity(cfg.robot_pose_port_to_unity)}")

        # -------------------------
        # (8) Initial camera pose: Windows -> Unity (Windows PUB, Unity SUB)
        # Windows binds, Unity connects
        # -------------------------
        self.initial_camera_pose_pub_to_unity = ctx.socket(zmq.PUB)
        self.initial_camera_pose_pub_to_unity.setsockopt(zmq.SNDHWM, 1)
        self.initial_camera_pose_pub_to_unity.setsockopt(zmq.LINGER, 0)
        self.initial_camera_pose_pub_to_unity.bind(cfg.to_unity(cfg.initial_camera_pose_port_to_unity))
        if self.verbose:
            print(f"[MainScene] InitialCameraPose PUB (to Unity) bound on {cfg.to_unity(cfg.initial_camera_pose_port_to_unity)}")

        # -------------------------
        # (9) Workspace bounds: Windows -> Unity (Windows PUB, Unity SUB)
        # Windows binds, Unity connects
        # -------------------------
        self.workspace_bounds_pub_to_unity = ctx.socket(zmq.PUB)
        self.workspace_bounds_pub_to_unity.setsockopt(zmq.SNDHWM, 1)
        self.workspace_bounds_pub_to_unity.setsockopt(zmq.LINGER, 0)
        self.workspace_bounds_pub_to_unity.bind(cfg.to_win(cfg.workspace_bounds_port_to_unity))
        if self.verbose:
            print(f"[MainScene] WorkspaceBounds PUB (to Unity) bound on {cfg.to_win(cfg.workspace_bounds_port_to_unity)}")

        # -------------------------
        # (10) Hand Tracking 1: Unity -> Windows (Unity PUB, Windows SUB)
        # Unity binds, Windows connects
        # -------------------------
        self.hand_sub1_from_unity = ctx.socket(zmq.SUB)
        self.hand_sub1_from_unity.setsockopt(zmq.CONFLATE, 1)
        self.hand_sub1_from_unity.setsockopt_string(zmq.SUBSCRIBE, "")
        self.hand_sub1_from_unity.connect(cfg.to_unity(cfg.hand1_port_from_unity))
        if self.verbose:
            print(f"[MainScene] Hand1 SUB (from Unity) connected to {cfg.to_unity(cfg.hand1_port_from_unity)}")

        # # -------------------------
        # # (11) Hand Tracking 2: Unity -> Windows (Unity PUB, Windows SUB)
        # # Unity binds, Windows connects
        # # -------------------------
        # self.hand_sub2_from_unity = ctx.socket(zmq.SUB)
        # self.hand_sub2_from_unity.setsockopt(zmq.CONFLATE, 1)
        # self.hand_sub2_from_unity.setsockopt_string(zmq.SUBSCRIBE, "")
        # self.hand_sub2_from_unity.connect(cfg.to_unity(cfg.hand2_port_from_unity))
        # if self.verbose:
        #     print(f"[MainScene] Hand2 SUB (from Unity) connected to {cfg.to_unity(cfg.hand2_port_from_unity)}")

        # -------------------------
        # (12) Hand forwarding: Windows -> Ubuntu (Windows PUB, Ubuntu SUB)
        # Windows binds, Ubuntu connects
        # -------------------------
        self.hand_pub_to_linux = ctx.socket(zmq.PUB)
        self.hand_pub_to_linux.setsockopt(zmq.SNDHWM, 1)
        self.hand_pub_to_linux.setsockopt(zmq.LINGER, 0)
        self.hand_pub_to_linux.bind(cfg.to_win(cfg.hand_pose_port_to_ubuntu))
        if self.verbose:
            print(f"[MainScene] Hand PUB (to Ubuntu) bound on {cfg.to_win(cfg.hand_pose_port_to_ubuntu)}")

    def set_tx_rate_hz(self, name: str, hz: float):
        self.tx_rate_hz[name] = float(hz) if hz is not None else 0.0
        self.last_tx_time.setdefault(name, 0.0)

    def set_rx_rate_hz(self, name: str, hz: float):
        self.rx_rate_hz[name] = float(hz) if hz is not None else 0.0
        self.last_rx_time.setdefault(name, 0.0)

    def should_tx(self, name: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        hz = float(self.tx_rate_hz.get(name, 0.0) or 0.0)
        if hz <= 0.0:
            return True  # no throttle
        interval = 1.0 / hz
        last = float(self.last_tx_time.get(name, 0.0))
        if (now - last) < interval:
            return False
        self.last_tx_time[name] = now
        return True

    def should_rx(self, name: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        hz = float(self.rx_rate_hz.get(name, 0.0) or 0.0)
        if hz <= 0.0:
            return True  # no throttle
        interval = 1.0 / hz
        last = float(self.last_rx_time.get(name, 0.0))
        if (now - last) < interval:
            return False
        self.last_rx_time[name] = now
        return True

    # ---------- COMMUNICATION FUNCTIONS ----------
    def receive_robot_status(self) -> Optional[dict]:
        """
        Receive robot status from Ubuntu (SUB), non-blocking only.
        Throttled by: rx_robot_status_from_ubuntu
        """
        if self.robot_status_sub_from_linux is None:
            return None

        if not self.should_rx("rx_robot_status_from_ubuntu"):
            return None

        try:
            return self.robot_status_sub_from_linux.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        except Exception:
            return None


    def publish_robot_pose_to_linux(self) -> bool:
        """
        Publish a calibrated robot pose to Ubuntu via PUB/SUB (fire-and-forget).
        Throttled by: tx_robot_pose_to_ubuntu

        Returns:
            True if published (or throttled-off), False if missing socket / bad pose.
        """
        if self.robot_pose_pub_to_linux is None:
            if self.verbose:
                print("[MainScene] robot_pose_pub_to_linux is None; did you call setup_communication()?")
            return False

        if not self.should_tx("tx_robot_pose_to_ubuntu"):
            return True  # throttled, but not an error

        # Pull latest calibrated pose
        try:
            T = self.T_robot_in_world
        except Exception:
            if self.verbose:
                print("[MainScene] T_robot_in_world missing in object_pose_estimator.extrinsics")
            return False

        if not isinstance(T, np.ndarray) or T.shape != (4, 4):
            if self.verbose:
                shp = getattr(T, "shape", None)
                print(f"[MainScene] publish_robot_pose_to_linux: expected 4x4 ndarray, got {type(T)} shape={shp}")
            return False

        msg = {
            "type": "calib_robot_pose",
            "timestamp": time.time(),
            "T": T.tolist(),
        }

        try:
            self.robot_pose_pub_to_linux.send_string(json.dumps(msg))
        except Exception as e:
            if self.verbose:
                print(f"[MainScene] Failed to publish robot pose: {repr(e)}")
            return False

        if self.verbose:
            # keep this quiet if it's frequent
            pass

        return True
    
    def send_object_poses(self):
        self.latest_object_poses = self.object_pose_estimator.get_latest_object_poses()

        self.objects_in_scene = []

        poses_for_unity = []
        poses_for_linux = []

        for obj_name, T in self.latest_object_poses.items():
            if np.all(T == 0):
                poses_for_unity.append({
                    "name": obj_name,
                    "pose": np.zeros(16).tolist(),
                    "visible": False
                })
                poses_for_linux.append({
                    "name": obj_name,
                    "pose": np.zeros(16).tolist(),
                    "visible": False
                })
                continue

            # Extract and convert pose
            R_obj = T[:3, :3]
            t_obj = T[:3, 3]

            q_obj = rotation_matrix_to_quaternion(R_obj)
            q_obj_wxyz = [q_obj[3], q_obj[0], q_obj[1], q_obj[2]]
            q_obj_unity = open3d_to_unity_quaternion(q_obj_wxyz)

            R_obj_unity = R.from_quat([q_obj_unity[1], q_obj_unity[2], q_obj_unity[3], q_obj_unity[0]]).as_matrix()

            R_y_90 = R.from_euler('y', 0, degrees=True).as_matrix()  # If needed
            R_obj_unity_aligned = R_y_90 @ R_obj_unity

            t_obj_unity = open3d_to_unity_vector(t_obj)

            T_obj_unity = np.identity(4)
            T_obj_unity[:3, :3] = R_obj_unity_aligned
            T_obj_unity[:3, 3] = t_obj_unity

            poses_for_unity.append({
                "name": obj_name,
                "pose": T_obj_unity.T.flatten().tolist(),
                "visible": True
            })
            poses_for_linux.append({
                "name": obj_name,
                "pose": T.flatten().tolist(),
                "visible": True
            })

            self.objects_in_scene.append(obj_name)

        # sorted_obj_names = sorted(self.object_pose_dict.keys())

        # # Red (GT), Green (predicted), Blue (optional/blank)
        top_targets = ["", "", ""]
        top_positions = [[0.0, 0.0, 0.0] for _ in range(3)]

        # if (self.intent_model_inference == True):
        #     # --- Green: Top-1 Prediction ---
        #     if self.top3_pred_indices and self.top3_pred_indices[0] is not None and self.top3_pred_indices[0] >= 0:
        #         idx = self.top3_pred_indices[0]
        #         if idx < len(sorted_obj_names):
        #             pred_name = sorted_obj_names[idx]
        #             if pred_name in self.object_pose_dict:
        #                 T, dims = self.object_pose_dict[pred_name]
        #                 local_top = np.array([dims[0] / 2, dims[1] / 2, dims[2], 1.0])
        #                 world_top = T @ local_top
        #                 top_targets[0] = pred_name
        #                 top_positions[0] = (open3d_to_unity_vector(world_top[:3]) + np.array([0, 0.15, 0])).tolist()


        # === Compose Unity message format ===
        unity_message = {
            "target": top_targets[0],
            "poses": poses_for_unity,
            "top_positions": top_positions,
            "interaction_state": "recording"
        }
        # print (poses_for_unity)
        if poses_for_unity:
            self.object_pose_pub_to_unity.send_string(json.dumps(unity_message))  # ✅ Send to Unity

        if poses_for_linux:
            self.object_pose_pub_to_linux.send_string(json.dumps(poses_for_linux))  # Original format for Linux

    def publish_robot_joint_angles(self) -> bool:
        """
        PUB joint angles to Unity.
        Returns True if sent this call, False otherwise.
        Throttle key: tx_robot_joints_unity
        """
        if self.robot_joint_pub_to_unity is None:
            return False
        if self.robot_status is None or self.robot_joint_angles_radians is None:
            return False

        if not self.should_tx("tx_robot_joints_unity"):
            return False

        # convert to degrees
        robot_joint_degrees = [math.degrees(x) for x in self.robot_joint_angles_radians]

        # apply any Unity-specific offset BEFORE creating message
        # (keep this here if Unity robot root expects it)
        # robot_joint_degrees[0] += 180.0

        joint_msg = {
            "timestamp": float(self.robot_status.get("timestamp", time.time())),
            "joint_values": robot_joint_degrees
        }

        try:
            self.robot_joint_pub_to_unity.send_string(json.dumps(joint_msg))
            return True
        except Exception as e:
            if self.verbose:
                print(f"[MainScene] Failed to publish joints: {repr(e)}")
            return False
    
    def gripper_value_to_angle(
        self,
        sensor_val: float,
        closed_val: float = 3.0,
        open_val: float = 230.0,
        closed_angle: float = 45.0,
        open_angle: float = 0.0
    ) -> float:
        # avoid divide-by-zero
        denom = (open_val - closed_val)
        if abs(denom) < 1e-9:
            return float(open_angle)

        # clamp sensor value
        sensor_val = max(min(sensor_val, open_val), closed_val)

        # alpha: 0=closed, 1=open
        alpha = (sensor_val - closed_val) / denom

        # linear interpolation
        return closed_angle * (1.0 - alpha) + open_angle * alpha
    
    def publish_gripper_state(self) -> bool:
        """
        PUB gripper angle to Unity.
        Returns True if sent this call, False otherwise.
        Throttle key: tx_gripper_state_unity
        """
        if self.gripper_pub_to_unity is None:
            return False
        if self.robot_status is None or self.gripper_status is None:
            return False

        if not self.should_tx("tx_gripper_state_unity"):
            return False

        # read actual gripper value (DON'T override)
        gripper_val = float(self.gripper_status.get("position", 0.0))

        # gripper_val = 233 #this is opened value

        # your calibration numbers
        closed_val = 3.0
        open_val = 233.0

        angle = self.gripper_value_to_angle(
            gripper_val,
            closed_val=closed_val,
            open_val=open_val,
            closed_angle=45.0,
            open_angle=0.0
        )
        self.gripper_angle = angle

        # # Prefer JSON for consistency (you can switch Unity to parse JSON)
        # msg = {
        #     "timestamp": float(self.robot_status.get("timestamp", time.time())),
        #     "gripper_position": gripper_val,
        #     "gripper_angle_deg": angle
        # }


        try:
            # self.gripper_pub_to_unity.send_string(json.dumps(msg))
            self.gripper_pub_to_unity.send_string(str(angle))
            return True
        except Exception as e:
            if self.verbose:
                print(f"[MainScene] Failed to publish gripper: {repr(e)}")
            return False

    def publish_initial_robot_pose_to_unity(self, one_shot: bool = True) -> bool:
        """
        Publish initial robot pose (Open3D world frame -> Unity) as:
            { "robot_matrix": [16 floats, column-major] }

        Returns:
            True  -> actually sent this call
            False -> skipped (throttled / one-shot already sent / missing data / error)
        """
        if self.initial_robot_pose_pub_to_unity is None:
            if self.verbose:
                print("[MainScene] initial_robot_pose_pub_to_unity is None; did you call setup_communication()?")
            return False

        # one-shot gate
        if one_shot and getattr(self, "sent_initial_robot_pose_to_unity", False):
            return False

        # frequency gate
        if not self.should_tx("tx_initial_robot_pose_unity"):
            return False

        # fetch T_robot_in_world
        if self.T_robot_in_world is None:
            if self.object_pose_estimator is not None and hasattr(self.object_pose_estimator, "extrinsics"):
                self.T_robot_in_world = self.object_pose_estimator.extrinsics.get("T_robot_in_world", None)

        T = self.T_robot_in_world
        if T is None:
            if self.verbose:
                print("[MainScene] ❌ T_robot_in_world is None.")
            return False
        if not isinstance(T, np.ndarray) or T.shape != (4, 4):
            if self.verbose:
                print(f"[MainScene] ❌ T_robot_in_world must be 4x4 np.ndarray, got {type(T)} shape={getattr(T,'shape',None)}")
            return False

        R_robot_o3d = T[:3, :3]
        t_robot_o3d = T[:3, 3]

        # Open3D -> Unity rotation via quaternion
        q_xyzw = R.from_matrix(R_robot_o3d).as_quat()  # [x,y,z,w]
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        q_unity_wxyz = open3d_to_unity_quaternion(q_wxyz)
        q_unity_xyzw = [q_unity_wxyz[1], q_unity_wxyz[2], q_unity_wxyz[3], q_unity_wxyz[0]]
        R_robot_unity = R.from_quat(q_unity_xyzw).as_matrix()

        # prefab local-axis correction (keep if needed)
        R_y_90 = R.from_euler('y', -90, degrees=True).as_matrix()
        R_robot_unity = R_robot_unity @ R_y_90

        t_robot_unity = open3d_to_unity_vector(t_robot_o3d)

        T_robot_unity = np.eye(4, dtype=float)
        T_robot_unity[:3, :3] = R_robot_unity
        T_robot_unity[:3, 3] = t_robot_unity

        msg = {"robot_matrix": T_robot_unity.T.flatten().tolist()}

        try:
            self.initial_robot_pose_pub_to_unity.send_string(json.dumps(msg))
        except Exception as e:
            if self.verbose:
                print(f"[MainScene] Failed to publish initial robot pose: {repr(e)}")
            return False

        self.sent_initial_robot_pose_to_unity = True
        return True


    def publish_initial_camera_pose_to_unity(
        self,
        cam_name: str = "realsense_front",
        one_shot: bool = True,
    ) -> bool:
        """
        Publish the camera pose to Unity as:
            { "cam_name": str, "cam_matrix": [16 floats, column-major] }

        Returns:
            True  -> actually sent this call
            False -> skipped (throttled / one-shot already sent / missing data / error)
        """
        if self.initial_camera_pose_pub_to_unity is None:
            if self.verbose:
                print("[MainScene] initial_camera_pose_pub_to_unity is None; did you call setup_communication()?")
            return False

        # one-shot gate
        if one_shot and getattr(self, "sent_initial_camera_pose_to_unity", False):
            return False

        # frequency gate
        if not self.should_tx("tx_initial_camera_pose_unity"):
            return False

        if not hasattr(self, "T_world_cam_dict") or cam_name not in self.T_world_cam_dict:
            if self.verbose:
                print(f"[MainScene] ❌ T_world_cam_dict missing or cam '{cam_name}' not found.")
            return False

        T_world_cam = self.T_world_cam_dict[cam_name]
        if not isinstance(T_world_cam, np.ndarray) or T_world_cam.shape != (4, 4):
            if self.verbose:
                print(
                    f"[MainScene] ❌ T_world_cam for '{cam_name}' must be 4x4 np.ndarray, "
                    f"got {type(T_world_cam)} shape={getattr(T_world_cam,'shape',None)}"
                )
            return False

        T_world_cam = np.linalg.inv(T_world_cam)

        R_cam_o3d = T_world_cam[:3, :3]
        t_cam_o3d = T_world_cam[:3, 3]

       
        # Use scipy directly (no ambiguity about ordering)
        q_xyzw = R.from_matrix(R_cam_o3d).as_quat()  # [x, y, z, w]
        q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # [w, x, y, z]

        # Open3D -> Unity quaternion conversion (your helper)
        q_unity_wxyz = open3d_to_unity_quaternion(q_wxyz)

        # Back to rotation matrix in Unity coords
        q_unity_xyzw = [q_unity_wxyz[1], q_unity_wxyz[2], q_unity_wxyz[3], q_unity_wxyz[0]]
        R_cam_unity = R.from_quat(q_unity_xyzw).as_matrix()
         # prefab local-axis correction (keep if needed)
        R_x_90 = R.from_euler('x', -90, degrees=True).as_matrix()
        R_y_90 = R.from_euler('y', -90, degrees=True).as_matrix()
        R_cam_unity = R_cam_unity @ R_x_90
        t_cam_unity = open3d_to_unity_vector(t_cam_o3d)

        T_cam_unity = np.eye(4, dtype=float)
        T_cam_unity[:3, :3] = R_cam_unity
        T_cam_unity[:3, 3] = t_cam_unity
        
        
        msg = {
            "cam_name": cam_name,
            "cam_matrix": T_cam_unity.T.flatten().tolist(),  # column-major for Unity Matrix4x4
        }

        try:
            self.initial_camera_pose_pub_to_unity.send_string(json.dumps(msg))
        except Exception as e:
            if self.verbose:
                print(f"[MainScene] Failed to publish initial camera pose: {repr(e)}")
            return False

        self.sent_initial_camera_pose_to_unity = True
        return True


    def receive_hand_data1_from_unity(self) -> None:
        """
        Receive REAL hand data from Unity (hand1).
        Throttle key: rx_hand_data1_from_unity
        Updates: self.real_hand_data
        """
        if self.hand_sub1_from_unity is None:
            return
        if not self.should_rx("rx_hand_data1_from_unity"):
            return

        latest_msg = None
        while True:
            try:
                latest_msg = self.hand_sub1_from_unity.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                if self.verbose:
                    print(f"[MainScene] hand1 recv error: {repr(e)}")
                return

        if latest_msg is None:
            return

        try:
            self.real_hand_data = json.loads(latest_msg)
        except Exception as e:
            if self.verbose:
                print(f"[MainScene] hand1 json parse error: {repr(e)}")
            self.real_hand_data = None
            return

        if self.real_hand_data and self.visualize:
            self.visualizer.update_hand_visuals(self.real_hand_data)
           

    def receive_hand_data2_from_unity(self) -> None:
        """
        Receive SYNTHETIC hand data from Unity (hand2).
        Throttle key: rx_hand_data2_from_unity
        Updates: self.synthetic_hand_data
        """
        if self.hand_sub2_from_unity is None:
            return
        if not self.should_rx("rx_hand_data2_from_unity"):
            return

        latest_msg = None
        while True:
            try:
                latest_msg = self.hand_sub2_from_unity.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                if self.verbose:
                    print(f"[MainScene] hand2 recv error: {repr(e)}")
                return

        if latest_msg is None:
            return

        try:
            self.synthetic_hand_data = json.loads(latest_msg)
        except Exception as e:
            if self.verbose:
                print(f"[MainScene] hand2 json parse error: {repr(e)}")
            self.synthetic_hand_data = None
            return

        if self.synthetic_hand_data and self.visualize:
            # synthetic hand visual (recommended: separate visual function)
            # If you don't have it yet, create it in Visualizer.
            #TODO:
            # self.visualizer.update_synth_hand_visuals(self.synthetic_hand_data)
            pass

    def receive_hand_data(self) -> None:
        """Convenience wrapper."""
        self.receive_hand_data1_from_unity()
        # self.receive_hand_data2_from_unity()



    # -------------------------
    # Placeholders / TODO functions
    # -------------------------
    def publish_workspace_bounds_to_unity(self) -> bool:
        return False


    def publish_object_poses(self) -> bool:
        return False

    def receive_object_pose_updates_from_unity(self) -> Optional[dict]:
       return None

    def publish_hand_to_linux(self, prefer: str = "real") -> bool:
        return False


    def close_communication(self):
        """Close all ZMQ sockets + context (matches the new socket member names)."""
        if self.verbose:
            print("[MainScene] Closing ZMQ communication...")

        # Collect every socket member you created in setup_communication()
        sockets = [
            # Ubuntu -> Windows
            self.robot_status_sub_from_linux,

            # Windows -> Ubuntu
            self.robot_pose_pub_to_linux,
            self.object_pose_pub_to_linux,
            self.hand_pub_to_linux,

            # Windows -> Unity
            self.object_pose_pub_to_unity,
            self.gripper_pub_to_unity,
            self.robot_joint_pub_to_unity,
            self.initial_robot_pose_pub_to_unity,
            self.initial_camera_pose_pub_to_unity,
            self.workspace_bounds_pub_to_unity,

            # Unity -> Windows
            self.object_pose_sub_from_unity,
            self.hand_sub1_from_unity,
            # self.hand_sub2_from_unity,
        ]

        for s in sockets:
            if s is not None:
                try:
                    # 0 linger = drop unsent messages and close immediately
                    s.close(0)
                except Exception:
                    pass

        # Null out references
        self.robot_status_sub_from_linux = None

        self.robot_pose_pub_to_linux = None
        self.object_pose_pub_to_linux = None
        self.hand_pub_to_linux = None

        self.object_pose_pub_to_unity = None
        self.gripper_pub_to_unity = None
        self.robot_joint_pub_to_unity = None
        self.initial_robot_pose_pub_to_unity = None
        self.initial_camera_pose_pub_to_unity = None
        self.workspace_bounds_pub_to_unity = None

        self.object_pose_sub_from_unity = None
        self.hand_sub1_from_unity = None
        # self.hand_sub2_from_unity = None

        # Terminate context last
        if self.zmq_context is not None:
            try:
                self.zmq_context.term()
            except Exception:
                pass
            self.zmq_context = None


    def run_loop(self, hz: float = 120.0):
        if self.verbose:
            print("[MainScene] Starting main loop... (Ctrl+C to stop)")

        hz = float(hz)
        dt = 1.0 / hz if hz > 0 else 0.0

        try:
            while True:
                # -------------------------------------------------
                # (1) RX: robot status (Ubuntu -> Windows)
                # -------------------------------------------------
                # print ("hello")
                if (self.connect_robot):
                    msg = self.build_robot_status_msg()
                else:
                    msg = self.receive_robot_status() 

                if msg is not None:
                    self.robot_status = msg

                    # safe getters (avoid crashes if fields missing)
                    self.gripper_status = self.robot_status.get("gripper", None)
                    self.robot_joint_angles_radians = self.robot_status.get("joints", None)
                    self.robot_joint_angles_radians[0] += math.pi

                    # update robot visuals only if we got joints
                    if self.visualize and self.robot_joint_angles_radians is not None:
                        self.visualizer.update_robot_visuals(self.robot_joint_angles_radians)

                # -------------------------------------------------
                # (2) RX: hands (Unity -> Windows)
                # -------------------------------------------------
                self.receive_hand_data1_from_unity()
                self.receive_hand_data2_from_unity()

                # -------------------------------------------------
                # (3) TX: robot joints + gripper (Windows -> Unity)
                # -------------------------------------------------
                self.publish_gripper_state()
                self.publish_robot_joint_angles()

                # -------------------------------------------------
                # (4) TX: initial poses (Windows -> Unity) (one-shot)
                # -------------------------------------------------
                self.publish_initial_robot_pose_to_unity(one_shot=False)
                self.publish_initial_camera_pose_to_unity("realsense_front", one_shot=False)

                # -------------------------------------------------
                # (5) TX: calibrated robot pose to Ubuntu (optional)
                # -------------------------------------------------
                if self.transfer_to_ubuntu and self.connect_robot:
                    self.publish_robot_pose_to_linux()

                # -------------------------------------------------
                # (6) Objects: visualize latest poses (and optionally publish)
                # -------------------------------------------------
                if self.real_object and self.object_pose_estimator is not None:
                    self.latest_object_poses = self.object_pose_estimator.get_latest_object_poses()
                    
                    if self.visualize and self.latest_object_poses is not None:
                        for obj in self.object_pose_estimator.objects:
                            T = self.latest_object_poses.get(obj.name, None)
                        
                            if T is not None:
                                self.visualizer.update_object_visual(obj, T)
                            

                    # If you later re-enable object pubs, do it here:
                    self.send_object_poses()
               
                # -------------------------------------------------
                # (7) Render / exit condition
                # -------------------------------------------------
                if self.visualize:
                    if not self.visualizer.vis.poll_events():
                        if self.verbose:
                            print("👋 Visualization window closed.")
                        break
                    self.visualizer.vis.update_renderer()

                if dt > 0:
                    time.sleep(dt)

        except KeyboardInterrupt:
            if self.verbose:
                print("[MainScene] Main loop interrupted by user.")
        finally:
            self.close_all()

    def close_all(self):
        """Cleanly close robot and any other resources."""
        if self.verbose:
            print("[MainScene] Closing all resources...")

        self.close_communication()

        #TODO:
        # if self.sensor_receiver is not None:
        #     self.sensor_receiver.close()
        # if self.object_pose_estimator is not None:
        #     self.object_pose_estimator.shutdown()
        # if self.visualizer is not None:
        #     self.visualizer.shutdown()
