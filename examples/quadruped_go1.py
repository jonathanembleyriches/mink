from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import typing
import keyboard  # Library for capturing keyboard inputs.

from typing import Optional
_HERE = Path(__file__).parent
_XML = _HERE / "unitree_go1" / "scene.xml"

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

    feet = ["FL", "FR", "RR", "RL"]

    base_task = mink.FrameTask(
        frame_name="trunk",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(model, cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    tasks = [base_task, posture_task, *feet_tasks]

    move_speed = 0.01  # Adjust as needed.
    target_position = np.array([0.5, 0.3, 0.2])  # Use a NumPy array for mutable storage.
    base_mid = model.body("trunk_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for foot in feet:
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, "trunk_target", "trunk", "body")

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
                target_name="FL_target",
                new_position=target_position,
            )
            # Update task targets.
            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))
            for i, task in enumerate(feet_tasks):
                task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

            # Compute velocity, integrate and set control signal.
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
