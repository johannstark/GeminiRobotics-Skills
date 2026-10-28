---
name: threejs-rendering
description: Renders MuJoCo simulations with Three.js (in-browser WASM or streamed Python). Use when setting up Z-up scenes, exporting compiled mesh manifests, per-frame body pose sync (wxyz to xyzw quaternions), streaming poses via WebSockets or robust polling with in-flight guards and reset epoch snapping, embedding 3D viewers in FastAPI or Gradio with no-store cache control, capturing snapshots/video for Gemini Robotics-ER, deprojecting ER [y, x] points to 3D, running headless Playwright viewer tests, or debugging offset, twisted, zero-size, or blank renders.
---

# Three.js rendering for MuJoCo

**Build static geometry once, then update only transforms each frame.** MuJoCo is Z-up, uses
half-sizes, Z-aligned cylinders and capsules, and **wxyz** quaternions. Three.js defaults to Y-up, full
sizes, Y-aligned cylinders and **xyzw** quaternions. Nearly every rendering bug comes from one of these.

## 1. Scene setup (Z-up, capture-ready)

```ts
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';   // path valid on 0.164 and 0.18x

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true }); // needed for toDataURL
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100);
camera.up.set(0, 0, 1);                       // MuJoCo is Z-up. With this, no axis conversion is needed anywhere.
camera.position.set(2, -1.5, 2.5);
const controls = new OrbitControls(camera, renderer.domElement);
const grid = new THREE.GridHelper(5, 50); grid.rotation.x = Math.PI / 2;   // lay the grid on XY
```

Set `camera.up` **before** creating `OrbitControls`, because the controls cache the up vector.

## 2. Choose an approach

| Approach | How | Use when |
|---|---|---|
| **A. One `Group` per body** (the Franka demo) | Each body group holds its geoms at their local `geom_pos`/`geom_quat`. Copy `xpos`/`xquat` each frame. | Default. Picking via `userData.bodyID`. Works with any pose source. |
| **B. `mjv_updateScene`** ([official demo](https://github.com/google-deepmind/mujoco/tree/main/wasm)) | Sync one pooled mesh per `scene.geoms` entry | You want MuJoCo's own decorations (contacts, tendons, sites) |
| **C. Remote Python** | Python steps, and streams `xpos`/`xquat` over WebSocket or polling. The browser runs A. | GPU, ROS, or real-robot bridges, heavy compute |

Geom conversion: `references/geom-conversion.md`. Python streaming & manifests: `references/remote-python.md`.
Snapshots, perception camera, video, and pixel↔3D: `references/perception-and-video.md`.
Tested scripts (`pip install mujoco playwright`, **run** them): `scripts/export_meshes.py` (`references/remote-python.md`) and `scripts/test_viewer_smoke.py` (`references/testing.md`).

## 3. Per-frame sync (approach A)

```ts
for (let i = 0; i < model.nbody; i++) {
  bodies[i].position.set(data.xpos[3*i], data.xpos[3*i+1], data.xpos[3*i+2]);
  bodies[i].quaternion.set(data.xquat[4*i+1], data.xquat[4*i+2], data.xquat[4*i+3], data.xquat[4*i]); // wxyz -> xyzw
}
renderer.render(scene, camera);
```

Loop order: inputs → `mj_step` (as many times as the sim-time budget allows) → sync → render
(see `MujocoSim.startLoop`). The per-geom alternative uses `data.geom_xpos` and `data.geom_xmat`
(row-major 3×3 → `Matrix4.set(r00,r01,r02,0, r10,…)`), with geometry at the identity pose.

## 4. ER points ↔ 3D

```ts
// ER [y, x] 0-1000 on the captured image -> ray from the SAME camera that captured it
const ndc = new THREE.Vector2((x / 1000) * 2 - 1, 1 - (y / 1000) * 2);
raycaster.setFromCamera(ndc, captureCamera);
const hit = raycaster.intersectObjects(simMeshes, false)[0];     // or raycaster.ray.intersectPlane(table, out)

// 3D -> ER-style [y, x] (to draw expectations or build prompts)
const v = p.clone().project(captureCamera);
const yx = [(1 - v.y) * 500, (v.x + 1) * 500];
```

This is valid for any aspect ratio **as long as the image is exactly the camera's full viewport** (no crop,
no letterbox) and the camera didn't move between capture and raycast. Skip hits on the robot or floor by
walking up to `userData.bodyID`. The hit is the top surface, so lower it to grasp height.

## 5. Gotchas

| Symptom | Cause |
|---|---|
| Robot lies sideways, orbit tumbles | `camera.up` not `(0,0,1)`, or set after `OrbitControls` was created |
| Links twisted | Missing wxyz→xyzw reorder |
| Boxes twice as big | Forgot that sizes are half-extents |
| Cylinders or capsules point the wrong way | Missing `geometry.rotateX(Math.PI/2)` |
| Blank or black snapshot | `preserveDrawingBuffer: false`, or the pixels were read in a different task than the render |
| Perception image darker than the screen | Render targets output **linear** colour (verified in three r186). Convert to sRGB (see the perception reference). |
| ER markers offset | Camera moved, `[y, x]` swapped, image cropped or letterboxed, or the device-pixel-ratio canvas was resized differently |
| Mat or workspace drawn with zero size or offset | JS recomputed extents from corner config with an ordering assumption; send extents from Python or use min/max. |
| Meshes offset or rotated vs physics | Loaded raw STLs instead of compiled `mesh_vert`/`mesh_face` (MuJoCo re-centers meshes; use `scripts/export_meshes.py`). |
| Memory grows on reload | `geometry.dispose()`, `material.dispose()`, `renderTarget.dispose()`, and delete the WASM model and data |
| Jank at high speed | Too many `mj_step` calls per frame. Cap the steps, or move physics to a Worker. |

Three.js version notes (r148 globals, r152 `colorSpace`): see `references/perception-and-video.md`.

## 6. In the Franka demo

The [Franka pick-and-place demo](https://ai.studio/apps/bundled/robotics_franka_pick_and_place) is the reference app these skills were built from. File names below refer to it.

- `RenderSystem.ts` uses `preserveDrawingBuffer: true`, a Z-up `PerspectiveCamera(45)`, damped `OrbitControls`,
  `simGroup` (one group per body, `userData.bodyID`), `erGroup` (bobbing cone markers), instanced contact markers,
  camera tweens (`moveCameraTo`), and `getCanvasSnapshot(w, h, mime)`.
- `project2DTo3D(x, y, camPos, lookAt)` clones the main camera, poses it at the capture viewpoint
  (`(0, -0.01, 2.0)` looking at the origin), and raycasts `simGroup` meshes.
- `rendering/GeomBuilder.ts` handles the geom→mesh factory. Group-3 geoms are skipped, planes become a `Reflector` floor,
  and mesh geoms read `mesh_vert`/`mesh_face`.
- `DragStateManager.ts` and `SelectionManager.ts` handle mouse picking and drag forces through `xfrc_applied`.
- The `index.html` import map pins three **0.164.1**, while `package.json` has `^0.181`. Keep `three/examples/jsm/...`
  imports, which work in both.
