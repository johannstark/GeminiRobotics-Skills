# MuJoCo in Python

Snippets were run with `mujoco` 3.16.0 on macOS (Python 3.12) and Menagerie `franka_emika_panda`.
`scripts/panda_pick_place.py` contains all of them in one runnable, tested file.

## Contents
- [Install and get robots](#install-and-get-robots)
- [Build scenes with MjSpec](#build-scenes-with-mjspec)
- [Step, real time, viewer](#step-real-time-viewer)
- [Named access and state](#named-access-and-state)
- [Contacts and grasp checks](#contacts-and-grasp-checks)
- [Cameras: RGB, depth, segmentation](#cameras-rgb-depth-segmentation)
- [Intrinsics and deprojection](#intrinsics-and-deprojection)
- [Recording episode video](#recording-episode-video)
- [Streaming to a browser](#streaming-to-a-browser)
- [MuJoCo inside a web server](#mujoco-inside-a-web-server)
- [The sim as an oracle for a real camera](#the-sim-as-an-oracle-for-a-real-camera)
- [Test harness patterns](#test-harness-patterns)
- [Troubleshooting](#troubleshooting)

## Install and get robots

```bash
pip install mujoco numpy pillow "imageio[ffmpeg]"
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
# or one robot only:
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set franka_emika_panda
# alternative: pip install robot_descriptions  (downloads on demand)
```

Menagerie robots ship `scene.xml` (robot, floor and light) and `<robot>.xml`. Many include a `home` keyframe.

## Build scenes with MjSpec

Edit models in code instead of patching XML strings:

```python
import mujoco, numpy as np

spec = mujoco.MjSpec.from_file("mujoco_menagerie/franka_emika_panda/scene.xml")
spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034])           # between the fingertips

cube = spec.worldbody.add_body(name="cube", pos=[0.45, 0.3, 0.02])
cube.add_freejoint()
cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02],   # HALF-sizes
              rgba=[0.9, 0.1, 0.1, 1], mass=0.05, friction=[2, 0.3, 0.1], condim=4)
tray = spec.worldbody.add_body(name="tray", pos=[0.45, -0.3, 0])
tray.add_geom(name="tray_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.005], rgba=[0.2, 0.2, 0.2, 1])

# Identity camera orientation looks straight down -Z, with image "up" = world +Y.
spec.worldbody.add_camera(name="top", pos=[0.45, 0.0, 1.4], fovy=60)

for key in spec.keys:                                     # keyframes must match the new nq
    key.qpos = list(key.qpos) + [0.45, 0.3, 0.02, 1, 0, 0, 0]

spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)
```

- Other loaders: `MjModel.from_xml_path(p)`, `MjModel.from_xml_string(xml, assets={"mesh.stl": bytes})`.
- `spec.to_xml()` dumps the edited model, which helps when debugging.
- Hand-written MJCF: angles default to **degrees**. Add `<compiler angle="radian"/>` if you think in radians.
- A camera that looks at a point: compute a rotation whose −Z points at the target and set `quat=`, or use
  `mode="targetbody" target="..."` in XML.

## Step, real time, viewer

```python
import time, mujoco.viewer
with mujoco.viewer.launch_passive(model, data) as viewer:    # macOS: run with `mjpython script.py`
    while viewer.is_running():
        t0 = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
```

- Headless: just call `mj_step` in a loop. It runs much faster than real time.
- Linux servers: `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU) **before** importing mujoco, for `Renderer`.
- On macOS, `Renderer` works without a display. It may warn `ARB_clip_control unavailable … depth accuracy will be
  limited`. This was harmless in testing, with millimetre-level deprojection.
- `mujoco.mj_rollout` / MJX handle batched rollouts.

## Named access and state

```python
data.body("cube").xpos          # (3,) world position — a VIEW, .copy() to keep
data.body("cube").xquat         # wxyz
data.site("tcp").xpos, data.site("tcp").xmat      # xmat: flattened row-major 3x3
data.joint("joint1").qpos, data.joint("joint1").qvel
model.joint("joint1").qposadr[0], model.joint("joint1").dofadr[0]   # model.jnt(...) also works
model.actuator("actuator8").ctrlrange              # Panda gripper: [0, 255]
model.key("home").id, model.camera("top").id
mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
```

Free joints use 7 qpos values (`x y z qw qx qy qz`) and 6 dof. To teleport, write qpos, zero the matching qvel,
then call `mj_forward`.

## Contacts and grasp checks

```python
def touching(model, data, body_a, body_b):
    a, b = model.body(body_a).id, model.body(body_b).id
    for i in range(data.ncon):
        c = data.contact[i]
        if {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} == {a, b}:
            return True
    return False

held = touching(model, data, "left_finger", "cube") and touching(model, data, "right_finger", "cube")
lifted = data.body("cube").xpos[2] > 0.05
```

Contact forces: `f = np.zeros(6); mujoco.mj_contactForce(model, data, i, f)`, in the contact frame.

## Cameras: RGB, depth, segmentation

```python
W, H = 640, 480
renderer = mujoco.Renderer(model, height=H, width=W)       # H, W must be <= offheight/offwidth

renderer.update_scene(data, camera="top"); rgb = renderer.render().copy()           # (H,W,3) uint8
renderer.enable_depth_rendering(); renderer.update_scene(data, camera="top")
depth = renderer.render().copy(); renderer.disable_depth_rendering()                 # (H,W) float32 metres
renderer.enable_segmentation_rendering(); renderer.update_scene(data, camera="top")
seg = renderer.render().copy(); renderer.disable_segmentation_rendering()            # (H,W,2) [obj_id, obj_type]
```

- Call `.copy()` on every render. The renderer reuses its buffer.
- `camera=-1` is the default free camera. Named or fixed cameras give reproducible perception.
- Segmentation pixels hold `[geom_id, mjtObj.mjOBJ_GEOM]` for geoms. Use them to check that an ER point lands on
  the expected object (`pixel_is_geom` in the script).

## Intrinsics and deprojection

MuJoCo cameras have a vertical `fovy`, square pixels and a centred principal point. The camera frame is x right,
y up, looking along −Z. This differs from OpenCV, where y points down and +Z is forward.

```python
def intrinsics(model, cam_id, W, H):
    f = 0.5 * H / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    return f, f, W / 2, H / 2

def pixel_to_world(model, data, cam, u, v, depth):
    cid = model.camera(cam).id
    fx, fy, cx, cy = intrinsics(model, cid, depth.shape[1], depth.shape[0])
    ui, vi = int(round(u)), int(round(v))
    d = float(np.median(depth[max(vi-2, 0):vi+3, max(ui-2, 0):ui+3]))      # robust to edges
    p_cam = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    return data.cam_xpos[cid] + data.cam_xmat[cid].reshape(3, 3) @ p_cam

def world_to_pixel(model, data, cam, p, W, H):
    cid = model.camera(cam).id
    fx, fy, cx, cy = intrinsics(model, cid, W, H)
    pc = data.cam_xmat[cid].reshape(3, 3).T @ (np.asarray(p) - data.cam_xpos[cid])
    return cx + fx * pc[0] / -pc[2], cy - fy * pc[1] / -pc[2]

def er_point_to_world(model, data, cam, yx, depth):        # Gemini Robotics-ER [y, x] 0-1000
    h, w = depth.shape
    return pixel_to_world(model, data, cam, yx[1] / 1000 * w, yx[0] / 1000 * h, depth)
```

Measured: the ER point for a 4 cm cube seen from 1.4 m deprojected to within about 5 mm of the cube's top face.
The 0–1000 integer grid is the main error source.

Export calibration for a real-robot-style pipeline: `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`, and
`T_world←cam_opencv = [R_mj @ diag(1,-1,-1) | t]`.

## Recording episode video

```python
import imageio.v3 as iio
rec = mujoco.Renderer(model, height=240, width=320)
frames = []
def record(d, cams=("top", -1)):
    views = []
    for c in cams:
        rec.update_scene(d, camera=c); views.append(rec.render().copy())
    frames.append(np.hstack(views))                   # multi-view grid like the ER notebook videos

# call record(data) every 1/fps seconds of sim time inside your control loop
iio.imwrite("episode.mp4", np.stack(frames), fps=10, codec="libx264")
```

When sending the video to ER, describe the grid ("each frame shows 2 views side by side: left = top camera,
right = free camera"). Hold the final pose for about 1 s before stopping, because ER grades the end state.

## Streaming to a browser

To render a Python sim in Three.js, stream `xpos`/`xquat` per body as Float32 over a WebSocket. See
`threejs-rendering/references/remote-python.md`. Body order must match on both sides. Load the same MJCF in the
browser (WASM, never stepped) only to build meshes, or send the geometry once.

## MuJoCo inside a web server

Rules for asynchronous backends (FastAPI, Gradio):
1. **One stepping owner**: Exactly one thread calls `mj_step`. Motion commands update `data.ctrl` and wait on time; never call `mj_step` from routes or controllers.
2. **Dedicated render thread**: Create `mujoco.Renderer` on its own thread; EGL contexts have OS thread affinity, so never lazy-render on the caller's thread. Split locks: `update_scene(data)` under physics lock, `render()` under render lock.
3. **No MuJoCo work in `async def`**: Never acquire physics locks or call MuJoCo functions inside `async def` handlers. Read pre-published atomic snapshots or offload via `asyncio.to_thread`.
4. **`MUJOCO_GL` before import**: Set `os.environ["MUJOCO_GL"] = "egl"` before importing `mujoco` anywhere in the app (test via subprocess, not runtime mutation).

## The sim as an oracle for a real camera

A simulated camera is an ideal ground-truth oracle for validating the real-world perception pipeline:

1. **Synthetic homography generation**:
   Compute the simulation camera homography by projecting known 3D mat corner coordinates $[X, Y, 0]$ into the
   camera image using `cam_xpos`, `cam_xmat`, and `cam_fovy`, with a $+0.5\text{ px}$ shift matching OpenCV's
   pixel convention.
2. **Camera pose test**:
   Feed the synthetic homography to `camera_from_homography` (from the `gemini-robotics-er` skill, `scripts/table_grounding.py`)
   and assert that the recovered optical center $C$ matches `data.cam_xpos` to within $5\text{ mm}$.
3. **ArUco textures and specular lighting**:
   Texture fiducial markers on $1\text{ mm}$ thin boxes with `specular="0"`. High specular glare blinds ArUco detection.
4. **Neutral floors**:
   Use a matte, neutral gray floor (`rgba="0.6 0.6 0.6 1"`). Colored or checkerboard textures cause HSV color
   detection false positives.
5. **Match camera FOV explicitly**:
   Set the simulation camera's vertical field of view to match the physical webcam's active horizontal FOV:
   $$fovy = 2 \cdot \arctan\left(\tan\left(\frac{hfov}{2}\right) \cdot \frac{H}{W}\right)$$
   Document clearly which perception constants transfer directly to reality and which are sim-only.

## Test harness patterns

Follow these patterns when writing offline simulation tests (see also the `gemini-robotics-er` skill, `testing.md`):
- **Seeded RNG**: Always seed random number generators (`np.random.default_rng(seed)`) to ensure identical object
  placement across test runs.
- **`slow` pytest marker**: Mark physics-heavy integration runs with `@pytest.mark.slow` so fast unit tests run quickly.
- **No-op `time.sleep`**: In test suites, patch the module's local `time.sleep` rather than global Python sleep.
- **Cube-spawn clearance**: Ensure initial cube placements have at least $2\text{ cm}$ clearance from the home jaw pose;
  otherwise, the cube can spawn inside the gripper collision volume.
- **Metric thresholds**: Assert all spatial tolerances in metric millimeters in the table plane.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `XML Error: ... could not open file` | Wrong `meshdir`/relative path. Load from the robot's folder, or use `assets=` |
| `keyframe ... invalid qpos size` | Extend `key.qpos` for added free joints |
| Render crash or blank on Linux | Set `MUJOCO_GL=egl` or `osmesa` before the first `import mujoco`; render size must be ≤ `offwidth`/`offheight` |
| `launch_passive` fails on macOS | Run with `mjpython` |
| Deprojected point floats mid-air | The ray hit the robot (occlusion). Check `seg`, or move the arm or objects |
| Arm doesn't move | Writing `qpos` instead of `ctrl`, wrong actuator index, or ctrl outside `ctrlrange` |
| Object slips or left behind during lift | Writing `qpos` teleports the arm with zero `qvel`, leaving the object behind. Command `ctrl`. Check pad friction (use ~1.8) and `forcerange` window (±1.5 N·m). |
| Cube squeezed out sideways | Gripper forcerange is too high (3.35 N·m). Reduce actuator `forcerange` to `[-1.5, 1.5]`. |
| Jaws cannot report "empty" aperture | Friction pads collide with each other. Add `<exclude body1="gripper" body2="moving_jaw_so101_v1"/>`. |
| Finger penetrates object on descent | TCP site defined at fixed jaw inner face rather than grasp midpoint. Re-derive TCP between pads. |
| Sim runs faster than real time | Multiple threads calling `mj_step` (e.g. frame loop + motion command). Establish one stepping owner. |
| `gladLoadGL error` over SSH | No X11 display found. Set `os.environ["MUJOCO_GL"] = "egl"` before `import mujoco`. |
| Explodes or NaN | Bodies overlap at t=0, tiny inertia, large timestep, or extreme gains |
| Joint limits seem 57× too tight | MJCF angles are in degrees by default. Add `<compiler angle="radian"/>`. |
