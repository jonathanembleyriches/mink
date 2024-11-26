
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import keyboard  # Library for capturing keyboard inputs.

from typing import Optional
import mink

_HERE = Path(__file__).parent
_XML = _HERE / "z1_arm.xml"  # Adjust the path to your XML file.

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
    target_id = model.body(target_name).mocapid[0]
    if new_position is not None:
        data.mocap_pos[target_id] = new_position
    elif offset is not None:
        current_pos = data.mocap_pos[target_id]
        data.mocap_pos[target_id] = current_pos + offset
    else:
        raise ValueError("Either new_position or offset must be provided.")

if __name__ == "__main__":
    # Load the MuJoCo model and data.
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Define initial target position and movement speed.
    target_name = "link06"  # Adjust if needed to match the target body.
    move_speed = 0.01  # Speed of movement per key press.
    target_position = np.array([0.5, 0.3, 0.2])  # Initial position.

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Rate limiter for controlling simulation update frequency.
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

            # Update the target position in the MuJoCo simulation.
            move_target(
                model=model,
                data=data,
                target_name=target_name,
                new_position=target_position,
            )

            # Step the simulation forward.
            mujoco.mj_step(model, data)

            # Update the viewer and maintain the simulation rate.
            viewer.sync()
            rate.sleep()
