import rospy
from sensor_msgs.msg import JointState
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import typing
# import keyboard  # Library for capturing keyboard inputs.

from typing import Optional
import mink
from scipy.spatial.transform import Rotation as R  # For quaternion manipulation.

_HERE = Path(__file__).parent
_XML = _HERE / "kuka_iiwa_14" / "scene.xml"

_XML = _HERE / "unitree_z1" / "scene.xml"
def move_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_name: str,
    new_position: Optional[np.ndarray] = None,
    new_rotation: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    rotation_offset: Optional[np.ndarray] = None,
) -> None:
    """
    Move the target location to a new position or adjust its orientation.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        target_name: The name of the target body or site to move.
        new_position: The new position (x, y, z) for the target. If specified, 
                      the target will be set to this position.
        new_rotation: A new quaternion (w, x, y, z) for the target. If specified,
                      the target's orientation will be set to this rotation.
        offset: An optional offset (dx, dy, dz) to apply to the current position.
        rotation_offset: An optional offset (roll, pitch, yaw) in radians to apply
                         to the current rotation.
    """
    target_id = model.body(target_name).mocapid[0]
    if new_position is not None:
        data.mocap_pos[target_id] = new_position
    elif offset is not None:
        current_pos = data.mocap_pos[target_id]
        data.mocap_pos[target_id] = current_pos + offset

    if new_rotation is not None:
        data.mocap_quat[target_id] = new_rotation
    elif rotation_offset is not None:
        current_quat = data.mocap_quat[target_id]
        current_rot = R.from_quat(current_quat)
        rotation_delta = R.from_euler("xyz", rotation_offset)
        new_rot = (current_rot * rotation_delta).as_quat()
        data.mocap_quat[target_id] = new_rot
def move_target_old(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_name: str,
    new_position: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
) -> None:
    """
    Move the target location to a new position or by an offset.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        target_name: The name of the target body or site to move.
        new_position: The new position (x, y, z) for the target. If specified, 
                      the target will be set to this position.
        offset: An optional offset (dx, dy, dz) to apply to the current position.
                Ignored if `new_position` is provided.
    """
    if new_position is not None:
        data.mocap_pos[model.body(target_name).mocapid[0]] = new_position
    elif offset is not None:
        current_pos = data.mocap_pos[model.body(target_name).mocapid[0]]
        data.mocap_pos[model.body(target_name).mocapid[0]] = current_pos + offset
    else:
        raise ValueError("Either new_position or offset must be provided.")

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model=model, cost=1e-2),
    ]

    ## =================== ##
    rospy.init_node("mujoco_joint_state_publisher", anonymous=True)
    joint_state_msg = JointState()
    joint_state_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    move_speed = 0.01  # Adjust as needed.
    target_position = np.array([0.5, 0.3, 0.2])  # Use a NumPy array for mutable storage.
    rotation_speed = 0.05
    target_rotation = np.array([1.0, 0.0, 0.0, 0.0]) 
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Update task target.

            # Handle keyboard inputs.
            # if keyboard.is_pressed("left"):
            #     target_position[0] -= move_speed
            # if keyboard.is_pressed("right"):
            #     target_position[0] += move_speed
            # if keyboard.is_pressed("up"):
            #     target_position[1] += move_speed
            # if keyboard.is_pressed("down"):
            #     target_position[1] -= move_speed
            # if keyboard.is_pressed("w"):  # Move up in z-axis
            #     target_position[2] += move_speed
            # if keyboard.is_pressed("s"):  # Move down in z-axis
            #     target_position[2] -= move_speed
            rotation_offset = np.zeros(3)
            # if keyboard.is_pressed("q"):  # Roll counter-clockwise
            #     rotation_offset[0] += rotation_speed
            # if keyboard.is_pressed("e"):  # Roll clockwise
            #     rotation_offset[0] -= rotation_speed
            # if keyboard.is_pressed("a"):  # Pitch down
            #     rotation_offset[1] += rotation_speed
            # if keyboard.is_pressed("d"):  # Pitch up
            #     rotation_offset[1] -= rotation_speed
            # if keyboard.is_pressed("z"):  # Yaw left
            #     rotation_offset[2] += rotation_speed
            # if keyboard.is_pressed("c"):  # Yaw right
            #     rotation_offset[2] -= rotation_speed

            # Update the target position and rotation in the MuJoCo simulation.
            move_target(
                model=model,
                data=data,
                target_name="target",
                new_position=target_position,
                rotation_offset=rotation_offset,
            )
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)
            joint_state_msg.name = [model.joint(i).name for i in range(model.njnt)]
            joint_state_msg.position = [data.qpos[model.jnt_qposadr[i]] for i in range(model.njnt)]
            joint_state_msg.velocity = [data.qvel[model.jnt_dofadr[i]] for i in range(model.njnt)]
            joint_state_msg.effort = [data.qfrc_applied[model.jnt_dofadr[i]] for i in range(model.njnt)]
            print(joint_state_msg)

            # Publish the JointState message
            joint_state_pub.publish(joint_state_msg)
            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
