from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import typing
import keyboard  # Library for capturing keyboard inputs.

from typing import Optional
import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_z1" / "scene.xml"

def move_target(
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

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (["link06"], ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    max_velocities = {
        "joint1": np.pi,
        "joint2": np.pi,
        "joint3": np.pi,
        "joint4": np.pi,
        "joint5": np.pi,
        "joint6": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    # Target movement speed per key press.
    move_speed = 0.01  # Adjust as needed.
    target_position = np.array([0.5, 0.3, 0.2])  # Use a NumPy array for mutable storage.
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0, warn=False)

        while viewer.is_running():
            # Handle keyboard inputs.
            if keyboard.is_pressed("left"):
                target_position[0] -= move_speed
            if keyboard.is_pressed("right"):
                target_position[0] += move_speed
            if keyboard.is_pressed("up"):
                target_position[1] += move_speed
            if keyboard.is_pressed("down"):
                target_position[1] -= move_speed
            if keyboard.is_pressed("w"):  # Move up in z-axis
                target_position[2] += move_speed
            if keyboard.is_pressed("s"):  # Move down in z-axis
                target_position[2] -= move_speed

            # Move target to the new position based on keyboard input.
            move_target(
                model=model,
                data=data,
                target_name="target",
                new_position=target_position,
            )

            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()

