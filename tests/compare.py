import mujoco
import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.mujoco import (
    load_robot_description as mj_load_description,
)
from robot_descriptions.loaders.pinocchio import (
    load_robot_description as pin_load_description,
)

"""
WORLD: refers to the origin of the world as if it was attached to the desired frame.
LOCAL_WORLD_ALIGNED: refers to the frame centered at the desired frame but with axes aligned with the WORLD axes.
"""


def skew(x):
    wx, wy, wz = x
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def adjoint(R, t):
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    A = np.block(
        [
            [R, np.zeros((3, 3))],
            [skew(t) @ R, R],
        ]
    )
    assert A.shape == (6, 6)
    return A


def inverse(R, t):
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


np.set_printoptions(precision=2, threshold=1e-5, suppress=True)

# PINNOCHIO
print("Loading pinnochio model...")
robot = pin_load_description("panda_description")
q = 0.1 * np.pi * (2.0 * np.random.random((9,)) - 1.0)
model_pin = robot.model
data_pin = robot.data
pin.computeJointJacobians(model_pin, data_pin, q)
pin.updateFramePlacements(model_pin, data_pin)
frame_id = model_pin.getFrameId("panda_link7")

# MUJOCO
print("Loading mujoco model...")
model_mj = mj_load_description("panda_mj_description")
data_mj = mujoco.MjData(model_mj)
data_mj.qpos = q
mujoco.mj_kinematics(model_mj, data_mj)
mujoco.mj_comPos(model_mj, data_mj)


# Check frame transforms are identical.
T_sb_pin = data_pin.oMf[frame_id].copy()
t_sb_pin = T_sb_pin.np[:3, 3]
R_sb_pin = T_sb_pin.np[:3, :3]
t_sb = data_mj.body("link7").xpos.copy()
R_sb = data_mj.body("link7").xmat.copy().reshape(3, 3)
np.testing.assert_allclose(t_sb, t_sb_pin, atol=1e-7)
np.testing.assert_allclose(R_sb, R_sb_pin, atol=1e-7)

# Check that mujoco's jac matches pinnochio's jac in frame: LOCAL_WORLD_ALIGNED.
jac_pin_local_world_aligned: np.ndarray = pin.getFrameJacobian(
    model_pin, data_pin, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)
jac_muj = np.zeros((6, model_mj.nv))
mujoco.mj_jacBody(
    model_mj, data_mj, jac_muj[:3], jac_muj[3:], model_mj.body("link7").id
)
np.testing.assert_allclose(jac_pin_local_world_aligned, jac_muj, atol=1e-7)

# Compute Jacobian in the local frame.
jac_pin_local: np.ndarray = pin.getFrameJacobian(
    model_pin, data_pin, frame_id, pin.ReferenceFrame.LOCAL
)
R_bs, t_bs = inverse(R_sb, t_sb)
A_bs = adjoint(R_bs, np.zeros(3))
jac_muj_local = A_bs @ jac_muj
np.testing.assert_allclose(jac_muj_local, jac_pin_local, atol=1e-7)
