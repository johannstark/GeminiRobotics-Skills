# Three.js Viewer Testing & Headless Verification

Browser-based 3D viewers often suffer from regressions that pass HTTP status checks but produce
blank canvases, corrupted coordinate frames, or missing meshes.

## Contents
- [Headless Playwright smoke test](#headless-playwright-smoke-test)
- [Three critical viewer assertions](#three-critical-viewer-assertions)
- [Running the smoke test](#running-the-smoke-test)
- [Testing state endpoints](#testing-state-endpoints)

## Headless Playwright smoke test

To catch regressions before deployment (such as raw STL offsets, zero-size mats, or broken WebGL shaders),
run a headless Playwright test against the actual rendered canvas.

See `scripts/test_viewer_smoke.py` for the complete implementation.

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
    page = browser.new_page()
    page.goto("http://127.0.0.1:8000/viewer")
    page.wait_for_function("window.viewerReady === true", timeout=10000)

    report = page.evaluate("window.getViewerReport()")
    browser.close()
```

## Three critical viewer assertions

1. **Canvas is not blank (pixel variance):**
   A black, transparent, or unshaded scene has near-zero pixel variance. Reading back RGBA buffer
   variance ensures geometry actually rendered with lighting.
   ```ts
   const gl = renderer.getContext();
   const pixels = new Uint8Array(width * height * 4);
   gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
   // Assert variance(pixels) > 10.0
   ```

2. **Mesh count matches geom count:**
   Iterate the Three.js scene graph to ensure `meshCount` matches the non-collision (`group !== 3`)
   geom count in `manifest.json`. If a mesh fails to load or an exporter assumes all group-2 geoms are meshes,
   this assertion flags the discrepancy immediately.

3. **Known body pose round-trips with non-trivial quaternion:**
   Set a known test body at `[0.1, 0.2, 0.3]` with 45° rotation about X (MuJoCo wxyz `[0.92388, 0.38268, 0, 0]`).
   Verify that Three.js reports position `[0.1, 0.2, 0.3]` and quaternion `[0.38268, 0, 0, 0.92388]` (xyzw),
   proving that wxyz→xyzw conversion is active.

## Running the smoke test

```bash
python threejs-rendering/scripts/test_viewer_smoke.py
```

The script:
1. Compiles a test MuJoCo model with a non-trivial pose and mixed geom types (mesh, box, sphere).
2. Generates the scene manifest and binary mesh files with `export_meshes.py`.
3. Serves the web viewer via a local ephemeral HTTP server.
4. Uses Chromium to verify canvas variance, mesh count, and quaternion order.

## Testing state endpoints

For remote Python setups (`references/remote-python.md`), verify the `/api/state` or WebSocket output schema:
- Must supply monotonic `seq` (uint32).
- Must supply `reset_epoch` (uint32).
- Must include $3 \times N$ positions and $4 \times N$ quaternions matching `model.nbody`.
- State endpoint must NEVER block on `mj_step` inside `async def` routes.
