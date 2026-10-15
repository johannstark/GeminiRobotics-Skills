---
name: mujoco-physics
description: Simulates robots with MuJoCo in the browser (WebAssembly, @mujoco/mujoco or mujoco-js) or natively in Python (mujoco, MjSpec). Use for loading MJCF / MuJoCo Menagerie robots and their assets, stepping physics, reading and writing qpos/ctrl/xpos, actuators and grippers, contacts, offscreen RGB/depth/segmentation cameras, pixel-to-3D deprojection for vision models, damped-least-squares inverse kinematics, waypoint pick-and-place, recording episode videos, and debugging unstable or mis-loaded models.
---

# MuJoCo physics (WASM and Python)

The concepts are identical in both runtimes. `MjModel` is static and compiled from MJCF. `MjData` is the
dynamic state. Call `mj_step` to advance time, and `mj_forward` to recompute derived quantities
(`xpos`, `site_xpos`, `cam_xpos`, contacts) without advancing.

Verified with `mujoco` 3.16 (Python), `@mujoco/mujoco` 3.16 and `mujoco-js` 0.0.7 (WASM, run under Bun),
and Menagerie `franka_emika_panda`.

## 1. Choose a runtime

| Runtime | Use when | Rendering | Reference |
|---|---|---|---|
| Python `mujoco` | Research, data generation, headless servers, notebooks, real-robot bridges, ER pipelines | `mujoco.Renderer` (RGB/depth/segmentation), `mujoco.viewer`, or stream poses to Three.js | `references/python.md` |
| WASM `@mujoco/mujoco` (official) | Zero-install web demos and UIs | Three.js (`threejs-rendering` skill) | `references/wasm.md` |
| WASM `mujoco-js` 0.0.7 (the Franka demo) | Existing code only | Three.js | `references/wasm.md` |
| MJX / MuJoCo Warp | Thousands of parallel envs on a GPU | Render a few with CPU MuJoCo | MuJoCo docs |

IK, control and pick-and-place: `references/ik-and-control.md`.
Tested end-to-end script: `scripts/panda_pick_place.py`. It builds the scene, renders RGB-D,
converts an ER `[y, x]` point to 3D, runs DLS IK and a pick-and-place, verifies with contacts, and records an MP4.
Run `python scripts/panda_pick_place.py --help`. It exits 0 when the cube ends up in the tray.

## 2. Ten rules

1. **Call `mj_forward` after any direct state edit** (`qpos`, reset, keyframe, mocap) before reading
   positions. `mj_step` does this internally.
2. **Command robots through `ctrl`**, not `qpos`. Writing `qpos` teleports the robot and fights the actuators.
   Respect `actuator_ctrlrange`.
3. **Check the gripper's range and direction**: the Menagerie Panda `actuator8` is `0`=closed and `255`=open (a
   tendon). Other grippers differ. Read `model.actuator(...).ctrlrange` and test it.
4. **Index spaces differ.** `qpos` uses `jnt_qposadr` (a free joint has 7 values: xyz + wxyz), while `qvel`, the
   Jacobian columns and `dq` use `jnt_dofadr` (a free joint has 6). Quaternions are **wxyz**.
5. **`xmat`, `site_xmat` and `cam_xmat` are row-major 3×3**, flattened to 9 values.
6. **Adding free bodies to a Menagerie scene breaks its keyframes** ("invalid qpos size"). Append
   `[x,y,z,1,0,0,0]` per free joint to every `key.qpos`, or drop the keyframe.
7. **Offscreen render size is capped** by `<visual><global offwidth offheight>` (default 640×480).
   Raise it before compiling.
8. **WASM memory is manual.** Call `.delete()` once on every `MjModel`, `MjData`, buffer, VFS, spec and
   `data.contact` copy. Typed arrays such as `data.qpos` are live views, so `.slice()` anything you keep.
9. **The perception camera must see the target.** In the Menagerie Panda `home` pose, the arm sits right
   under an overhead camera. Place objects to the side, move the arm away before capturing, or
   check segmentation (this happened during testing).
10. **Instability** (explosions, NaN, jitter) usually comes from interpenetrating bodies at start, tiny
    masses or inertias, a timestep that's too large, or very stiff gains. Fix the model before tuning the controller.

## 3. Core loop

```python
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")      # or MjSpec.from_file(...).compile()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)
while data.time < 5.0:
    data.ctrl[:] = controller(model, data)              # your policy or IK writes ctrl
    mujoco.mj_step(model, data)
```

```ts
import loadMujoco from '@mujoco/mujoco';
const mujoco: any = await loadMujoco();
const model = mujoco.MjModel.mj_loadXML('/working/scene.xml');   // files must be in mujoco.FS (or pass an MjVFS)
const data = new mujoco.MjData(model);
mujoco.mj_forward(model, data);
function frame() {                                               // fixed sim time per display frame
  const t0 = data.time;
  while (data.time - t0 < 1 / 60) mujoco.mj_step(model, data);
  syncThreeJs(data); requestAnimationFrame(frame);
}
// on teardown: data.delete(); model.delete();
```

## 4. WASM ↔ Python cheat sheet

| Need | Python | WASM (`@mujoco/mujoco` 3.16) |
|---|---|---|
| Load | `MjModel.from_xml_path(p)`, `from_xml_string(xml, assets={...})`, `MjSpec.from_file(p).compile()` | `MjModel.mj_loadXML(path)` (FS), `mj_loadXML(name, vfs)`, `from_xml_string(xml)`. mujoco-js: `MjModel.loadFromXML(path)` |
| Name → id | `model.body("x").id`, `mj_name2id(m, mjtObj.mjOBJ_BODY, "x")` | `mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY.value, 'x')`, `model.body('x').id`, `model.site('x')`, `model.jnt('x')` (no `model.joint`) |
| Body pose | `data.body("x").xpos`, `.xquat` (wxyz) | `data.xpos[3*i..]`, `data.xquat[4*i..]` |
| Site pose | `data.site("tcp").xpos`, `.xmat` | `data.site_xpos[3*i..]`, `data.site_xmat[9*i..]` |
| Joint | `data.joint("j").qpos`, `model.jnt_qposadr`, `model.jnt_dofadr` | same arrays |
| Control | `data.ctrl[a]`, `model.actuator_ctrlrange[a]` | `data.ctrl[a]`, `model.actuator_ctrlrange[2*a..]` |
| Contacts | `data.ncon`, `data.contact[i].geom1/geom2` | `const c = data.contact; c.get(i); c.delete()` (a **copy**) |
| Jacobian | `mj_jacSite(m, d, jacp(3,nv), jacr(3,nv), sid)` with numpy | `new mujoco.DoubleBuffer(3*nv)` → `buf.GetView()` → `buf.delete()` |
| Enums | `mujoco.mjtGeom.mjGEOM_BOX` | `mujoco.mjtGeom.mjGEOM_BOX.value` |
| Memory | Garbage collected | `.delete()` everything you create |

## 5. Bridging to Gemini Robotics-ER

ER returns `[y, x]` normalized 0–1000 on the image you sent. To lift it to 3D in Python, render depth from the
**same camera at the same moment**, then deproject:

```python
from panda_pick_place import render_rgbd, er_point_to_world, pixel_is_geom
rgb, depth, seg = render_rgbd(renderer, data, "top")       # send rgb to ER
p_surface = er_point_to_world(model, data, "top", [y, x], depth)
grasp = p_surface - [0, 0, half_height]                    # depth hits the top face
assert pixel_is_geom(model, seg, [y, x], "cube_geom")      # occlusion / wrong-object check
```

In testing, the round trip world → ER point → world was accurate to about 5 mm at 1.4 m camera height,
which is limited by the 0–1000 integer grid. For the browser, raycast instead (`threejs-rendering` skill).
For ER calls, tool loops and video verification, see the `gemini-robotics-er` skill.

## 6. In the Franka demo

The [Franka pick-and-place demo](https://ai.studio/apps/bundled/robotics_franka_pick_and_place) is the reference app these skills were built from. File names below refer to it.

- `MujocoSim.ts` is the orchestrator. It loads `mujoco-js` (aliased `mujoco_wasm`, with the `.wasm` file from unpkg), runs a
  `requestAnimationFrame` loop that steps `1/60 × speed` seconds of sim time per frame, applies drag forces through
  `xfrc_applied`, and calls `dispose()` to delete the model, data and option and unmount `/working`.
- `RobotLoader.ts` fetches Menagerie files from GitHub raw into `/working`. It follows `file=` attributes and
  respects `meshdir`/`texturedir`. It injects 20 cubes (4 cm, 0.05 kg, friction 1.5, `condim=4`) and a
  `stack_base` tray, adds `<site name="tcp" pos="0 0 0.1">` to `hand`, and renames `actuator8` to `gripper`.
- `IkSystem.ts` and `FrankaAnalyticalIK.ts` provide closed-form Panda IK. They sweep q7 and pick the minimum of
  `|q−q_cur|² + 0.05|q−q_neutral|²`. This works for the Panda only. Use DLS for other arms.
- `SequenceAnimator.ts` is a 15-step pick/place state machine. It interpolates `ctrl[0..6]` with smoothstep
  and sets the gripper to 0/255. It stacks on a 3×3 grid (6 cm pitch). Targets come from ER raycasts with z−2 cm.
- `utils/StringUtils.ts::getName` reads names from `model.names` (mujoco-js predates named access, although
  `mj_name2id` exists in 0.0.7 too).
