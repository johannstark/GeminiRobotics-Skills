# Perception images, video and pixel ↔ 3D for Gemini Robotics-ER

## Contents
- [Snapshot of the visible canvas (the Franka demo)](#snapshot-of-the-visible-canvas-the-franka-demo)
- [Dedicated perception camera (recommended)](#dedicated-perception-camera-recommended)
- [Mapping ER output to 3D](#mapping-er-output-to-3d)
- [Recording canvas video](#recording-canvas-video)
- [Debug overlays](#debug-overlays)

## Snapshot of the visible canvas (the Franka demo)

`App.tsx::handleErSend`:

1. Tween the main camera to a fixed top view, `(0, -0.01, 2.0)` looking at the origin.
2. Wait 100 ms, then grab the canvas scaled to at most 640 px. The PNG is sent to ER.
3. Tween back, then raycast every result with a camera **cloned at the capture pose**.

```ts
renderer.render(scene, camera);                                    // render immediately before reading
const s = Math.min(640 / canvas.width, 640 / canvas.height, 1);    // keep the aspect ratio
const tmp = document.createElement('canvas');
tmp.width = Math.round(canvas.width * s); tmp.height = Math.round(canvas.height * s);
tmp.getContext('2d')!.drawImage(canvas, 0, 0, tmp.width, tmp.height);
const pngB64 = tmp.toDataURL('image/png').split(',')[1];           // Interactions API
const jpgB64 = tmp.toDataURL('image/jpeg', 0.8).split(',')[1];     // Live API frames (JPEG, ≤1 FPS)
```

Requires `preserveDrawingBuffer: true`, or call `toDataURL` in the same task right after `render`.
The drawbacks: the user's view jumps, and the image aspect ratio follows the window. Prefer a fixed camera.

## Dedicated perception camera (recommended)

A fixed camera rendered off-screen gives reproducible images, a constant aspect ratio, and the same
pose for capture and back-projection.

```ts
export class PerceptionCamera {
  readonly camera: THREE.PerspectiveCamera;
  private rt: THREE.WebGLRenderTarget;
  private static LUT = (() => {                       // linear -> sRGB, 8-bit
    const t = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) { const c = i / 255;
      t[i] = Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055)); }
    return t;
  })();

  constructor(private renderer: THREE.WebGLRenderer, private scene: THREE.Scene,
              readonly width = 768, readonly height = 576, fovY = 60) {
    this.camera = new THREE.PerspectiveCamera(fovY, width / height, 0.05, 10);
    this.camera.up.set(0, 0, 1);
    this.rt = new THREE.WebGLRenderTarget(width, height);
  }

  setPose(pos: THREE.Vector3, lookAt: THREE.Vector3) {
    this.camera.position.copy(pos); this.camera.lookAt(lookAt); this.camera.updateMatrixWorld();
  }

  /** Returns base64 (no data: prefix). Pixels match the on-screen colours (no tone mapping). */
  capture(mime: 'image/png' | 'image/jpeg' = 'image/png', quality = 0.85): string {
    const { width: w, height: h } = this;
    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.rt);
    this.renderer.render(this.scene, this.camera);
    const buf = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(this.rt, 0, 0, w, h, buf);
    this.renderer.setRenderTarget(prev);

    const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d')!, img = ctx.createImageData(w, h), L = PerceptionCamera.LUT;
    for (let y = 0; y < h; y++) {                                   // GL rows are bottom-up: flip
      const src = (h - 1 - y) * w * 4, dst = y * w * 4;
      for (let x = 0; x < w * 4; x += 4) {
        img.data[dst + x] = L[buf[src + x]]; img.data[dst + x + 1] = L[buf[src + x + 1]];
        img.data[dst + x + 2] = L[buf[src + x + 2]]; img.data[dst + x + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL(mime, quality).split(',')[1];
  }

  dispose() { this.rt.dispose(); }
}
```

Why the lookup table: when rendering into a normal `WebGLRenderTarget`, three.js writes the **linear working colour
space** and skips `outputColorSpace` and tone mapping (verified in r186's `WebGLRenderer`). Without conversion,
ER sees a darker image than you do. An alternative is a second `WebGLRenderer` on its own canvas,
which applies sRGB output normally, at the cost of duplicate GPU uploads.

Tips:

- Place the camera so the robot doesn't occlude the workspace. Move the arm out of view before capturing, or
  use two cameras.
- About 800 px wide matches the ER notebook. For Live, use 768×768 JPEG at ≤1 FPS.
- To mirror the Python pipeline, render a depth target too (`rt.depthTexture = new THREE.DepthTexture(w, h)`),
  then linearize with near/far. Raycasting against meshes is usually simpler in the browser.

## Mapping ER output to 3D

```ts
function erPointTo3D(yx: [number, number], cam: THREE.PerspectiveCamera, meshes: THREE.Object3D[],
                     ignoreBodies = new Set<number>()) {
  const rc = new THREE.Raycaster();
  rc.setFromCamera(new THREE.Vector2((yx[1] / 1000) * 2 - 1, 1 - (yx[0] / 1000) * 2), cam);
  for (const hit of rc.intersectObjects(meshes, true)) {
    let o: THREE.Object3D | null = hit.object;
    while (o && o.userData.bodyID === undefined) o = o.parent;
    const body = o?.userData.bodyID ?? -1;
    if (!ignoreBodies.has(body)) return { point: hit.point.clone(), bodyId: body };
  }
  return null;
}

function boxCenter([ymin, xmin, ymax, xmax]: number[]): [number, number] {
  return [(ymin + ymax) / 2, (xmin + xmax) / 2];
}

function tablePlaneHit(yx: [number, number], cam: THREE.PerspectiveCamera, zTable = 0) {   // no meshes needed
  const rc = new THREE.Raycaster();
  rc.setFromCamera(new THREE.Vector2((yx[1] / 1000) * 2 - 1, 1 - (yx[0] / 1000) * 2), cam);
  return rc.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -zTable), new THREE.Vector3());
}
```

- Ignore robot links and the floor (`ignoreBodies`) so markers land on objects.
- Lower the hit to grasp height (the Franka demo uses −2 cm for 4 cm cubes, in `SequenceAnimator.prepareStep`).
- With `mj_ray` in WASM you can raycast against the physics geometry instead (`mujoco-physics/references/wasm.md`).

## Recording canvas video

Use this for ER moment finding and progress classification:

```ts
export function recordCanvas(canvas: HTMLCanvasElement, fps = 10) {
  const mime = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'].find(t => MediaRecorder.isTypeSupported(t))!;
  const rec = new MediaRecorder(canvas.captureStream(fps), { mimeType: mime, videoBitsPerSecond: 2_500_000 });
  const chunks: Blob[] = [];
  rec.ondataavailable = e => e.data.size && chunks.push(e.data);
  rec.start(1000);
  return { stop: () => new Promise<Blob>(res => { rec.onstop = () => res(new Blob(chunks, { type: mime.split(';')[0] })); rec.stop(); }) };
}
// const r = recordCanvas(renderer.domElement); ...run the pickup...; const blob = await r.stop();
// upload on your server: client.files.upload({ file: blob, config: { mimeType: 'video/webm' } })
```

- `captureStream` only captures frames that are rendered, so keep rendering while recording, and hold the final pose for about 1 s.
- To record the perception view instead of the user view, draw each `PerceptionCamera` capture onto a canvas and record that.
- Keep the API key server-side. Send the blob to your backend, which uploads it to the Files API.

## Debug overlays

- Draw ER results over the captured image with an SVG `viewBox="0 0 1000 1000" preserveAspectRatio="none"`. Points
  map to `cx = x`, `cy = y` directly. This is what `App.tsx::LogOverlay` does.
- In 3D, use small cones or spheres at hits (`RenderSystem.addErMarker`), and a `Line` for trajectories.
- Show the capture camera's frustum (`THREE.CameraHelper`) while developing, to catch pose mismatches.
