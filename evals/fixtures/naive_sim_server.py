import mujoco
from fastapi import FastAPI

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
app = FastAPI()

@app.get("/state")
async def get_state():
    mujoco.mj_step(model, data)
    return {"time": float(data.time), "qpos": data.qpos.tolist()}
