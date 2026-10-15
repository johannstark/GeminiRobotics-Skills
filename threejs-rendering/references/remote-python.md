# Render a Python MuJoCo simulation in the browser

Python owns the physics (native speed, GPU, ROS or ER agents). The browser only draws. The server below was
tested with `mujoco` 3.16 and `websockets` 15. Two clients each received 84 floats per frame for the Panda
(12 bodies × 7).

## Server (one physics loop, broadcast to every client)

```python
# server.py   pip install mujoco websockets numpy      python server.py path/to/scene.xml
import asyncio, json, sys, mujoco, numpy as np, websockets

model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
clients: set = set()
HEADER = json.dumps({"nbody": model.nbody, "bodies": [model.body(i).name for i in range(model.nbody)]})

async def handler(ws):
    await ws.send(HEADER)                         # client checks body order against its own model
    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)

async def physics(hz=60):
    loop = asyncio.get_running_loop()
    while True:
        t0, sim_t0 = loop.time(), data.time
        while data.time - sim_t0 < 1 / hz:
            mujoco.mj_step(model, data)           # your controller / ER agent writes data.ctrl here
        frame = np.concatenate([data.xpos.ravel(), data.xquat.ravel()]).astype(np.float32).tobytes()
        websockets.broadcast(clients, frame)      # drops slow clients' frames instead of blocking
        await asyncio.sleep(max(0, 1 / hz - (loop.time() - t0)))

async def main(port=8765):
    async with websockets.serve(handler, "0.0.0.0", port):
        await physics()

if __name__ == "__main__":
    asyncio.run(main())
```

Don't step physics inside `handler`. With per-connection loops, two open tabs would advance the sim twice as fast.

## Client

```ts
const ws = new WebSocket('ws://localhost:8765');
ws.binaryType = 'arraybuffer';
ws.onmessage = (e) => {
  if (typeof e.data === 'string') {                        // header
    const hdr = JSON.parse(e.data);
    if (hdr.nbody !== bodies.length) console.error('body count mismatch', hdr.nbody, bodies.length);
    return;
  }
  const f = new Float32Array(e.data), nb = bodies.length;
  for (let i = 0; i < nb; i++) {
    bodies[i].position.set(f[3 * i], f[3 * i + 1], f[3 * i + 2]);
    const q = 3 * nb + 4 * i;
    bodies[i].quaternion.set(f[q + 1], f[q + 2], f[q + 3], f[q]);          // wxyz -> xyzw
  }
};
```

## Getting the meshes to the browser

- **Easiest:** load the *same* MJCF in the browser with WASM (`mujoco-physics/references/wasm.md`), build meshes with
  `buildBodies` (`geom-conversion.md`), and **never step it**. Body order then matches, so verify it with the header names.
- **No WASM:** send geometry once as JSON or binary (`geom_type`, `geom_size`, `geom_pos`, `geom_quat`, `geom_rgba`,
  `geom_bodyid`, `mesh_vert`, `mesh_face`, and addresses) and run the same factory on plain arrays.
- Send commands back over the same socket (JSON), for example `{"type": "goal", "yx": [y, x]}` from an ER click.

## Alternatives

- Python-only: skip the browser, and use `mujoco.Renderer` for perception images and `mujoco.viewer` for interaction.
- Many bodies or high rates: send only the dynamic bodies, quantize to Float16, or send at 30 Hz and interpolate.
- Per-geom streaming (`geom_xpos` 3 + `geom_xmat` 9 floats per geom) works too. It is 12 floats per geom instead of
  7 per body, but needs no body hierarchy on the client.
