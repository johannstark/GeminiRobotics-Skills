# Render a Python MuJoCo simulation in the browser

Python owns the physics (native speed, GPU, ROS, or ER agents), while the browser only renders.

## Contents
- [Getting the meshes to the browser](#getting-the-meshes-to-the-browser)
- [Client factory consuming manifest and binary meshes](#client-factory-consuming-manifest-and-binary-meshes)
- [Transport: WebSocket default](#transport-websocket-default)
- [Transport: Polling fallback](#transport-polling-fallback)
- [Body smoothing: frame-rate independent exponential filter](#body-smoothing-frame-rate-independent-exponential-filter)
- [Embedding inside FastAPI + Gradio](#embedding-inside-fastapi--gradio)
- [Drawing ER 3D target markers](#drawing-er-3d-target-markers)

## Getting the meshes to the browser

MuJoCo re-centers and re-orients meshes at compile time. Raw STLs from asset folders will not line up with body transforms. Always use compiled `mesh_vert`/`mesh_face`, branch strictly on `geom_type` (not visual geom groups), and cache by model hash.

Run the exporter (`scripts/export_meshes.py`):
```bash
python scripts/export_meshes.py scene.xml --out-dir static/mesh_cache
```

## Client factory consuming manifest and binary meshes

```ts
import * as THREE from 'three';

export interface SceneManifest {
  model_hash: string;
  nbody: number;
  bodies: { id: number; name: string; parent_id: number }[];
  geoms: {
    id: number;
    name: string;
    body_id: number;
    type: number;
    type_name: string;
    size: number[];
    pos: number[];
    quat: number[]; // wxyz
    rgba: number[];
    group: number;
    mesh_id: number | null;
  }[];
  meshes: {
    id: number;
    bin_file: string;
    nvert: number;
    nface: number;
    vert_offset: number;
    face_offset: number;
  }[];
  perception_camera?: { pos: number[]; quat: number[]; fovy: number };
}

export async function buildSceneFromManifest(
  manifestUrl: string,
  scene: THREE.Scene
): Promise<{ bodies: THREE.Group[]; manifest: SceneManifest }> {
  const resp = await fetch(manifestUrl);
  const manifest: SceneManifest = await resp.json();
  const baseUrl = manifestUrl.substring(0, manifestUrl.lastIndexOf('/') + 1);

  // 1. Create a THREE.Group per body
  const bodies: THREE.Group[] = manifest.bodies.map(b => {
    const g = new THREE.Group();
    g.name = b.name;
    g.userData.bodyID = b.id;
    scene.add(g);
    return g;
  });

  // 2. Pre-fetch binary meshes
  const meshGeometries = new Map<number, THREE.BufferGeometry>();
  for (const m of manifest.meshes) {
    const binResp = await fetch(`${baseUrl}${m.bin_file}`);
    const buf = await binResp.arrayBuffer();
    // 8B magic + 4B nvert + 4B nface
    const nvert = m.nvert, nface = m.nface;
    const verts = new Float32Array(buf, m.vert_offset, nvert * 3);
    const faces = new Uint32Array(buf, m.face_offset, nface * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    geo.setIndex(new THREE.BufferAttribute(faces, 1));
    geo.computeVertexNormals();
    meshGeometries.set(m.id, geo);
  }

  // 3. Construct geoms and attach to their parent body
  for (const geom of manifest.geoms) {
    if (geom.group === 3) continue; // Skip collision-only geoms
    let geo: THREE.BufferGeometry | null = null;
    const s = geom.size;

    switch (geom.type_name) {
      case 'plane':
        geo = new THREE.PlaneGeometry(2 * s[0] || 5, 2 * s[1] || 5);
        break;
      case 'box':
        geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
        break;
      case 'sphere':
        geo = new THREE.SphereGeometry(s[0], 24, 16);
        break;
      case 'cylinder':
        geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2);
        break;
      case 'capsule':
        geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2);
        break;
      case 'mesh':
        if (geom.mesh_id !== null && meshGeometries.has(geom.mesh_id)) {
          geo = meshGeometries.get(geom.mesh_id)!;
        }
        break;
    }

    if (!geo) continue;

    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]),
      opacity: geom.rgba[3],
      transparent: geom.rgba[3] < 1.0,
      roughness: 0.4,
      metalness: 0.1,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(geom.pos[0], geom.pos[1], geom.pos[2]);
    // MuJoCo wxyz -> Three.js xyzw
    mesh.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
    mesh.userData.geomID = geom.id;
    mesh.userData.bodyID = geom.body_id;

    bodies[geom.body_id].add(mesh);
  }

  return { bodies, manifest };
}
```

## Transport: WebSocket default

WebSockets provide low-latency, ordered binary state updates with negligible protocol overhead.

### Server (one physics loop, broadcast snapshot)

```python
import asyncio, json, struct, threading, time, mujoco, numpy as np, websockets

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
physics_lock = threading.Lock()
clients: set = set()
latest_snapshot: bytes = b""
reset_epoch: int = 0
seq_counter: int = 0

def physics_thread():
    global latest_snapshot, seq_counter
    timestep = model.opt.timestep
    while True:
        t0 = time.perf_counter()
        with physics_lock:
            mujoco.mj_step(model, data)
            # Binary frame format: uint32 seq, uint32 reset_epoch, float32 poses
            seq_counter += 1
            header = struct.pack("<II", seq_counter, reset_epoch)
            poses = np.concatenate([data.xpos.ravel(), data.xquat.ravel()]).astype(np.float32).tobytes()
            latest_snapshot = header + poses
        # Settle sim frequency
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, timestep - elapsed))

async def broadcast_loop(hz=60):
    while True:
        if clients and latest_snapshot:
            websockets.broadcast(clients, latest_snapshot)
        await asyncio.sleep(1.0 / hz)

async def ws_handler(ws):
    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)
```

> [!IMPORTANT]
> **Single Stepping Owner:** Never call `mj_step` inside WebSocket handlers or `async def` routes. Multiple
> concurrent steppers cause simulation time to outrun real time. The dedicated physics thread steps; network
> handlers only read pre-serialized snapshots.

## Transport: Polling fallback

WebSockets are the default. If reverse proxies or firewall restrictions require HTTP polling, use an in-flight guard, monotonic sequence numbers with `reset_epoch`/`session_id` snap on restart or teleport, and a connection freshness badge:

```ts
let inFlight = false;
let lastSeq = -1, lastEpoch = -1, lastSessionId: string | null = null, lastSuccessTime = Date.now();

async function pollState(bodies: THREE.Group[]) {
  if (inFlight) return;
  inFlight = true;

  try {
    const res = await fetch('./api/state', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    lastSuccessTime = Date.now();
    updateConnectionBadge(true);

    const snap = data.reset_epoch !== lastEpoch || (data.session_id && data.session_id !== lastSessionId);
    // Discard stale out-of-order frame ONLY within the same epoch/session
    if (!snap && data.seq <= lastSeq) return;

    lastSeq = data.seq;
    lastEpoch = data.reset_epoch;
    if (data.session_id) lastSessionId = data.session_id;

    applyBodyPoses(bodies, data.xpos, data.xquat, snap);
  } catch (err) {
    if (Date.now() - lastSuccessTime > 1500) {
      updateConnectionBadge(false); // Disconnected
    }
  } finally {
    inFlight = false;
  }
}

setInterval(() => pollState(bodies), 33); // 30 Hz polling
```

## Body smoothing: frame-rate independent exponential filter

Never smooth individual geoms independently—their relative transforms are fixed, and per-geom lerping
drifts links apart during rapid movement. Smooth at the **body** level using exponential decay:

$$\alpha = 1 - \exp(-dt / \tau)$$

Where $\tau$ is the time constant:
- $\tau = 0.030\text{ s}$ (30 ms) filters network jitter without visual lag.
- When `snap === true` (on `reset_epoch` change), set $\alpha = 1.0$ to snap immediately without sliding across the scene.

```ts
const TAU_SECONDS = 0.030; // 30ms time constant bridges 30 Hz network packets to 60 FPS display

export function smoothBodyPose(
  body: THREE.Group,
  targetPos: THREE.Vector3,
  targetQuat: THREE.Quaternion,
  dt: number,
  snap: boolean = false
) {
  if (snap) {
    body.position.copy(targetPos);
    body.quaternion.copy(targetQuat);
    return;
  }
  const alpha = 1.0 - Math.exp(-dt / TAU_SECONDS);
  body.position.lerp(targetPos, alpha);
  body.quaternion.slerp(targetQuat, alpha);
}
```

## Embedding inside FastAPI + Gradio

When embedding the Three.js viewer inside a FastAPI backend alongside Gradio:

1. **Serve the viewer with `Cache-Control: no-store`:**
   Prevents iframes from caching stale scene manifests or JavaScript builds after model changes.
2. **Mount Gradio LAST at `/`:**
   FastAPI matches routes in declaration order. If Gradio is mounted at `/` first, it intercepts all API and viewer requests.
3. **Use relative URLs (`./api/state`, `./static/...`):**
   Ensures the app functions correctly when hosted behind reverse proxies or nested under subpaths (`root_path`).
4. **MJPEG camera stream vs `gr.Image`:**
   Stream camera frames over a dedicated route (`multipart/x-mixed-replace`). Polling `gr.Image` over HTTP freezes when round-trip latency exceeds frame duration.

```python
import asyncio, io
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

app = FastAPI()

# 1. Static assets and meshes with no-store cache control
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

def get_jpeg_frame() -> bytes:  # placeholder: replace with your camera_render_thread.get_jpeg_frame()
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (40, 120, 200)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()

@app.get("/viewer")
def get_viewer():
    p = Path("templates/viewer.html")
    html = p.read_text("utf-8") if p.exists() else "<!doctype html><title>Viewer</title>"
    return HTMLResponse(html, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

# 2. MJPEG streaming response for live camera feed
@app.get("/video_feed")
async def video_feed(request: Request):
    async def frame_generator():
        while not await request.is_disconnected():
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + get_jpeg_frame() + b"\r\n"
            await asyncio.sleep(0.033)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# 3. Mount Gradio app LAST at root
with gr.Blocks() as demo:
    gr.HTML('<iframe src="./viewer" style="width: 100%; height: 600px; border: none;"></iframe>')

app = gr.mount_gradio_app(app, demo, path="/")
```

## Drawing ER 3D target markers

When Gemini Robotics-ER returns a 2D workspace target, Python lifts it to 3D on the table plane and
publishes it to the viewer:

```ts
export class TargetMarker {
  readonly group = new THREE.Group();
  private sphere: THREE.Mesh;
  private stem: THREE.Line;

  constructor() {
    const geo = new THREE.SphereGeometry(0.015, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: false });
    this.sphere = new THREE.Mesh(geo, mat);

    const lineMat = new THREE.LineBasicMaterial({ color: 0x00e5ff });
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, 1),
    ]);
    this.stem = new THREE.Line(lineGeo, lineMat);

    this.group.add(this.sphere);
    this.group.add(this.stem);
    this.group.visible = false;
  }

  setTarget(x: number, y: number, z: number, tableZ: number = 0.0) {
    this.group.visible = true;
    this.sphere.position.set(x, y, z);
    // Draw vertical stem down to table surface
    const positions = new Float32Array([x, y, tableZ, x, y, z]);
    this.stem.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  }
}
```
