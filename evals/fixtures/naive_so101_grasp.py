import mujoco

def naive_grasp(model, data, q_grasp):
    data.qpos[:5] = q_grasp
    data.qpos[5] = 0.0
    mujoco.mj_forward(model, data)
    data.qpos[1] -= 0.3
    mujoco.mj_forward(model, data)
