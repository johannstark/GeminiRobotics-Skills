# MuJoCo in the browser (WebAssembly)

Verified by running `@mujoco/mujoco` 3.16.0 and `mujoco-js` 0.0.7 under Bun with the Menagerie Panda.
Where the package README and the shipped `.d.ts` disagree, this file follows the runtime.

## Contents
- [Bindings](#bindings)
- [Loading the module](#loading-the-module)
- [Getting files in: FS, MjVFS, strings](#getting-files-in)
- [Fetching a Menagerie robot](#fetching-a-menagerie-robot)
- [Stepping and the render loop](#stepping-and-the-render-loop)
- [State, names, control](#state-names-control)
- [Out-parameters and buffers](#out-parameters-and-buffers)
- [Contacts and rays](#contacts-and-rays)
- [Memory](#memory)
- [Gotchas](#gotchas)

## Bindings

| | `@mujoco/mujoco` (official, Google DeepMind) | `mujoco-js` 0.0.7 (the Franka demo) |
|---|---|---|
| Version | Tracks MuJoCo releases (3.16.0 = MuJoCo 3.16) | Old community build |
| Load | `MjModel.mj_loadXML(path)`, `mj_loadXML(name, vfs)`, `from_xml_string(xml)`, `from_xml_path` | `MjModel.loadFromXML(path)` |
| Named access | `model.body('x')`, `model.geom('x')`, `model.site('x')`, `model.jnt('x')`, `model.actuator('x')`. `data.site('x')` etc. **There is no `model.joint()`.** | None. Scan `model.names` (see `utils/StringUtils.ts`) or use `mj_name2id` |
| Also has | `MjSpec` (`parseXMLString`, `mj_compile`), `MjVFS`, `mj_jacSite`, `mj_ray`, `mjv_updateScene`, `DoubleBuffer`, a multi-threaded build `@mujoco/mujoco/mt` | `mj_name2id`, `mj_jacSite`, `mj_ray`, `MjSpec`, `MjVFS`, `DoubleBuffer`, `FS` |
| Status | WIP. Main APIs are well tested. `mjspec` bindings are less exercised. | Unmaintained. Prefer the official package for new work. |

## Loading the module

```ts
import loadMujoco from '@mujoco/mujoco';                       // MT build: '@mujoco/mujoco/mt' (needs COOP/COEP headers)
const mujoco: any = await loadMujoco({
  // Only if your bundler doesn't serve the .wasm next to mujoco.js:
  // locateFile: (p: string) => p.endsWith('.wasm') ? '/mujoco.wasm' : p,
  printErr: (t: string) => console.error('[mujoco]', t),       // XML errors and "Aborted" show up here
});
```

- Load the module **once** per page, and keep it in a module variable or a React ref.
- Multi-threaded build: the server must send `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`. In Vite, set them under `server.headers`.
- Vite: `import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url'` and `locateFile: () => wasmUrl` if the
  default resolution fails.
- **In the Franka demo** (`App.tsx`): `import loadMujoco from 'mujoco_wasm'` (a Vite alias to
  `mujoco-js/dist/mujoco_wasm.js`), with the `.wasm` fetched from `https://unpkg.com/mujoco-js@0.0.7/dist/mujoco_wasm.wasm`.
  `printErr` watches for `"Aborted"` to show the crash UI.

## Getting files in

MuJoCo's C core reads files, not URLs. **Every XML, mesh and texture must be present before you compile.**

| Option | When | Code |
|---|---|---|
| Single XML string | Primitives only | `mujoco.MjModel.from_xml_string(xml)` |
| Emscripten FS | Robots with assets (the Franka demo) | `FS.mkdir('/working')`, `FS.writeFile('/working/scene.xml', text)`, `FS.writeFile(path, uint8)`, then `MjModel.mj_loadXML('/working/scene.xml')` |
| `MjVFS` | Keep assets off the global FS | `const vfs = new mujoco.MjVFS(); vfs.addBuffer('scene.xml', bytes); ...; MjModel.mj_loadXML('scene.xml', vfs); vfs.delete()` |
| `MjSpec` | Edit programmatically | `const spec = mujoco.parseXMLString(xml /*, vfs */); const model = mujoco.mj_compile(spec /*, vfs */); spec.delete()` |

`FS.mkdir` throws if the directory exists, so wrap it in try/catch, and create directories one level at a time.

## Fetching a Menagerie robot

This is the algorithm from `RobotLoader.ts`, and it works for any Menagerie model:

```ts
async function loadMenagerie(mujoco: any, robot = 'franka_emika_panda', scene = 'scene.xml',
                             patch: (file: string, xml: string) => string = (_, x) => x) {
  const base = `https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/${robot}/`;
  try { mujoco.FS.unmount('/working'); } catch {}
  try { mujoco.FS.mkdir('/working'); } catch {}
  const seen = new Set<string>(), queue = [scene], parser = new DOMParser();
  while (queue.length) {
    const f = queue.shift()!; if (seen.has(f)) continue; seen.add(f);
    const res = await fetch(base + f); if (!res.ok) throw new Error(`${f}: ${res.status}`);
    let dir = '/working'; for (const p of f.split('/').slice(0, -1)) { dir += '/' + p; try { mujoco.FS.mkdir(dir); } catch {} }
    if (!f.endsWith('.xml')) { mujoco.FS.writeFile('/working/' + f, new Uint8Array(await res.arrayBuffer())); continue; }
    const xml = patch(f, await res.text());
    mujoco.FS.writeFile('/working/' + f, xml);
    const doc = parser.parseFromString(xml, 'text/xml'), comp = doc.querySelector('compiler');
    const meshdir = comp?.getAttribute('meshdir') ?? '', texdir = comp?.getAttribute('texturedir') ?? '';
    const cur = f.includes('/') ? f.slice(0, f.lastIndexOf('/') + 1) : '';
    doc.querySelectorAll('[file]').forEach(el => {
      const tag = el.tagName.toLowerCase();
      const pre = tag === 'mesh' ? meshdir : ['texture', 'hfield'].includes(tag) ? texdir : '';
      const norm: string[] = [];
      for (const s of (cur + (pre ? pre + '/' : '') + el.getAttribute('file')).split('/'))
        s === '..' ? norm.pop() : s && s !== '.' && norm.push(s);
      queue.push(norm.join('/'));
    });
  }
  return mujoco.MjModel.mj_loadXML('/working/' + scene);
}
```

Patch the XML in `patch` to add objects, a TCP site or a camera, as the Franka demo does. A typical stable
grasp object is a 4 cm box with `mass="0.05" friction="1.5 0.3 0.1" condim="4"`. To add a TCP site:
`<site name="tcp" pos="0 0 0.1034"/>` inside `hand` (the Franka demo uses 0.1).

> Fetching per file is slow on first load (the Panda has about 70 files). Cache responses (Cache API or
> IndexedDB), or vendor the files into `/public` for production.

## Stepping and the render loop

```ts
function frame() {
  if (!paused) {
    applyInputs(data);                                   // ctrl, xfrc_applied, mocap
    const t0 = data.time;
    while (data.time - t0 < (1 / 60) * speed) mujoco.mj_step(model, data);   // sim time per display frame
  }
  syncThreeJs(model, data);                              // body xpos/xquat -> meshes (threejs-rendering skill)
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
```

- Cap the work per frame. If a frame takes longer than its budget, drop sim time rather than spiralling.
- For heavy scenes, run physics in a **Web Worker** and post `xpos`/`xquat` (Float64Array, transferable) to the main thread.
- Reset: `mj_resetData(model, data)` or `mj_resetDataKeyframe(model, data, keyId)`, then `mj_forward`.

## State, names, control

```ts
const OBJ = mujoco.mjtObj;
const siteId = mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, 'tcp');        // -1 if missing
const bodyId = model.body('cube').id;                                         // official bindings
const tcp = data.site_xpos.subarray(3 * siteId, 3 * siteId + 3);              // live view
const R = data.site_xmat.subarray(9 * siteId, 9 * siteId + 9);                // row-major 3x3
const q = data.xquat.subarray(4 * bodyId, 4 * bodyId + 4);                    // w, x, y, z
const keep = Float64Array.from(tcp);                                          // copy if you store it

// Teleport a free body (qpos: x y z qw qx qy qz), then recompute:
const jid = model.body_jntadr[bodyId], a = model.jnt_qposadr[jid];
data.qpos.set([0.5, 0.1, 0.02, 1, 0, 0, 0], a);
mujoco.mj_forward(model, data);

// Drive actuators within range:
const [lo, hi] = [model.actuator_ctrlrange[2 * i], model.actuator_ctrlrange[2 * i + 1]];
data.ctrl[i] = Math.min(hi, Math.max(lo, target));

// External force/torque on a body (world frame), e.g. mouse drag (MujocoSim.applyDragForce):
data.xfrc_applied.set([fx, fy, fz, tx, ty, tz], 6 * bodyId);
```

Enums are objects in both bindings, so use `.value` (`mujoco.mjtGeom.mjGEOM_BOX.value === 6`). Code that must handle
both a number and an object can normalize with `(e as any)?.value ?? e`.

## Out-parameters and buffers

Functions that write into `mjtNum*` need a buffer. **In 3.16 the runtime API is `new DoubleBuffer(n)`
(a size) or `DoubleBuffer.FromArray(arr)`, and you read it with `.GetView()` (capital G).** The README's
`new DoubleBuffer([0,0,0])` and `.getView()` fail at runtime.

```ts
const nv = model.nv;
const jacp = new mujoco.DoubleBuffer(3 * nv), jacr = new mujoco.DoubleBuffer(3 * nv);
try {
  mujoco.mj_jacSite(model, data, jacp, jacr, siteId);
  const Jp = jacp.GetView();                 // Float64Array, row-major 3 x nv: Jp[row * nv + dof]
} finally { jacp.delete(); jacr.delete(); }

const quat = new mujoco.DoubleBuffer(4);
mujoco.mju_mat2Quat(quat, Array.from(R));    // array inputs accept plain number[]
```

Full DLS IK in TypeScript (tested): see `ik-and-control.md`.

## Contacts and rays

```ts
const contacts = data.contact;               // a COPY (std::vector proxy), stale after the next step
for (let i = 0; i < contacts.size(); i++) {
  const c = contacts.get(i);                 // c.geom1, c.geom2, c.dist, c.pos, c.frame
  if (new Set([model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]]).has(cubeBody)) { /* ... */ }
}
contacts.delete();

// Ray cast against MuJoCo geometry (3.16: 9 args, the last is the hit normal)
const geomid = new mujoco.IntBuffer(1), normal = new mujoco.DoubleBuffer(3);
const dist = mujoco.mj_ray(model, data, [0, 0, 0.8], [0, 0, -1], null, true, -1, geomid, normal);  // -1 = miss
geomid.delete(); normal.delete();
```

`mj_ray` is a physics-accurate alternative to Three.js raycasting for ER points. Build the ray from the
capture camera's pose and intrinsics.

## Memory

Every Embind handle lives on the WASM heap and is **not** garbage collected. Call `.delete()` exactly once on:
`MjModel`, `MjData`, `MjvOption`, `MjvScene`, `MjvCamera`, `MjvPerturb`, `MjSpec`, `MjVFS`, every `*Buffer`,
and every `data.contact` copy. Use `try/finally` around temporary buffers. Unmount or clean `/working` on
teardown (see `MujocoSim.dispose()`). Double-deleting throws a `BindingError`.

## Gotchas

- **MJCF angles default to degrees.** `<joint range="-1 1">` means ±1°, unless `<compiler angle="radian">`.
  Menagerie sets radians, so hand-written XML is where this bites. Actuator `ctrlrange` is **not**
  converted, which can leave position actuators commanding far beyond the joint limits.
- `"Aborted()"` in `printErr` usually means a bad model, an out-of-range index, or a call on a deleted object.
  Reload the module afterwards, because its state is corrupt.
- Typed arrays are views. Growing WASM memory can detach old views, so re-read `data.qpos` each frame instead of
  caching it across model reloads.
- The `index.html` import map in the Franka demo pins Three.js 0.164.1 for AI Studio, while `package.json` uses `^0.181`.
  Keep imports valid for both.
- Windows support for `@mujoco/mujoco` is experimental (per its README).
