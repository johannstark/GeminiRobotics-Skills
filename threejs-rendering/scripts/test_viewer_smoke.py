#!/usr/bin/env python3
"""Headless Playwright smoke test for the Three.js viewer and manifest exporter.

Verifies:
1. Canvas is not blank (pixel variance above threshold).
2. Rendered mesh count equals the scene manifest's visual geom count.
3. Known body's world pose round-trips correctly, including non-trivial quaternion rotation.
"""

from __future__ import annotations

import argparse
import http.server
import json
import math
import os
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import time
from typing import Any

import mujoco
import numpy as np

# Minimum pixel variance threshold to assert that the canvas rendered visual geometry rather than solid black/white
MIN_PIXEL_VARIANCE: float = 10.0

# Render viewport dimensions for testing
VIEW_WIDTH: int = 400  # Canvas pixel width
VIEW_HEIGHT: int = 300  # Canvas pixel height

# Non-trivial test rotation: 45 degrees about world X axis
ROT_ANGLE_RAD: float = math.pi / 4.0

# Candidate chrome browser paths in linux environments
CHROME_CANDIDATE_PATHS: tuple[str, ...] = (
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
)


def find_browser_executable() -> str | None:
    """Locates a viable Chromium or Chrome executable path."""
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    for p in CHROME_CANDIDATE_PATHS:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    return None


def run_viewer_smoke_test() -> None:
    """Executes the Playwright headless smoke test against a compiled MuJoCo scene."""
    chrome_path = find_browser_executable()
    if not chrome_path:
        print("[ViewerSmokeTest] Chromium executable not found on system. Skipping Playwright execution.")
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[ViewerSmokeTest] Playwright not installed. Skipping Playwright execution.")
        return

    from export_meshes import export_model

    # 1. Create a synthetic scene with a rotated body containing multiple geom types
    # 45 deg about X: wxyz = [cos(pi/8), sin(pi/8), 0, 0] = [0.9238795, 0.3826834, 0, 0]
    qx = math.sin(ROT_ANGLE_RAD / 2.0)
    qw = math.cos(ROT_ANGLE_RAD / 2.0)
    test_quat_str = f"{qw:.6f} {qx:.6f} 0 0"

    xml = f"""
    <mujoco model="viewer_smoke_scene">
      <asset>
        <mesh name="tetra" vertex="0 0 0  0.08 0 0  0 0.08 0  0 0 0.08" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
      </asset>
      <worldbody>
        <light pos="0 0 2"/>
        <camera name="perception" pos="0 -0.8 0.6" fovy="45"/>
        <geom name="floor" type="plane" size="1 1 0.1" rgba="0.7 0.7 0.7 1" group="0"/>
        <body name="target_body" pos="0.1 0.2 0.3" quat="{test_quat_str}">
          <geom name="mesh_part" type="mesh" mesh="tetra" rgba="1 0.2 0.2 1" group="2"/>
          <geom name="box_part" type="box" size="0.03 0.03 0.03" pos="0.1 0 0" rgba="0.2 1 0.2 1" group="2"/>
          <geom name="sphere_part" type="sphere" size="0.025" pos="-0.1 0 0" rgba="0.2 0.4 1 1" group="2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        man_path, manifest = export_model(model, out_dir=tmp_path)
        model_hash = manifest["model_hash"]
        manifest_rel_url = f"./{model_hash}/manifest.json"

        # Count expected visual geoms (group != 3)
        expected_mesh_count = sum(1 for g in manifest["geoms"] if g["group"] != 3)

        # Generate HTML test page embedding the client factory
        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Viewer Smoke Test</title>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js"
      }}
    }}
  </script>
</head>
<body style="margin: 0; background: #111;">
  <canvas id="c" width="{VIEW_WIDTH}" height="{VIEW_HEIGHT}"></canvas>
  <script type="module">
    import * as THREE from "three";

    const canvas = document.getElementById("c");
    const renderer = new THREE.WebGLRenderer({{ canvas, preserveDrawingBuffer: true, antialias: true }});
    renderer.setSize({VIEW_WIDTH}, {VIEW_HEIGHT});
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x151515);

    const camera = new THREE.PerspectiveCamera(50, {VIEW_WIDTH} / {VIEW_HEIGHT}, 0.01, 10);
    camera.position.set(0, -0.8, 0.6);
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0.2);

    const light = new THREE.DirectionalLight(0xffffff, 2.0);
    light.position.set(1, -1, 2);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));

    async function init() {{
      const resp = await fetch("{manifest_rel_url}");
      const man = await resp.json();
      const baseUrl = "./" + man.model_hash + "/";

      const bodies = man.bodies.map(b => {{
        const g = new THREE.Group();
        g.name = b.name;
        scene.add(g);
        return g;
      }});

      const meshMap = new Map();
      for (const m of man.meshes) {{
        const binResp = await fetch(baseUrl + m.bin_file);
        const buf = await binResp.arrayBuffer();
        const verts = new Float32Array(buf, m.vert_offset, m.nvert * 3);
        const faces = new Uint32Array(buf, m.face_offset, m.nface * 3);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geo.setIndex(new THREE.BufferAttribute(faces, 1));
        geo.computeVertexNormals();
        meshMap.set(m.id, geo);
      }}

      for (const g of man.geoms) {{
        if (g.group === 3) continue;
        let geo = null;
        const s = g.size;
        if (g.type_name === 'plane') geo = new THREE.PlaneGeometry(2 * s[0] || 5, 2 * s[1] || 5);
        else if (g.type_name === 'box') geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
        else if (g.type_name === 'sphere') geo = new THREE.SphereGeometry(s[0], 16, 16);
        else if (g.type_name === 'mesh' && g.mesh_id !== null) geo = meshMap.get(g.mesh_id);

        if (!geo) continue;
        const mat = new THREE.MeshStandardMaterial({{ color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]) }});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
        mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
        bodies[g.body_id].add(mesh);
      }}

      // Pose target body (body 1) in world coordinates matching model definition
      bodies[1].position.set(0.1, 0.2, 0.3);
      // wxyz -> xyzw
      bodies[1].quaternion.set({qx:.6f}, 0.0, 0.0, {qw:.6f});

      renderer.render(scene, camera);

      window.getReport = () => {{
        let meshCount = 0;
        scene.traverse(obj => {{ if (obj.isMesh) meshCount++; }});
        const gl = renderer.getContext();
        const pixels = new Uint8Array({VIEW_WIDTH} * {VIEW_HEIGHT} * 4);
        gl.readPixels(0, 0, {VIEW_WIDTH}, {VIEW_HEIGHT}, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

        let sum = 0, sumSq = 0, n = {VIEW_WIDTH} * {VIEW_HEIGHT} * 3;
        for (let i = 0; i < pixels.length; i += 4) {{
          sum += pixels[i] + pixels[i+1] + pixels[i+2];
          sumSq += pixels[i]*pixels[i] + pixels[i+1]*pixels[i+1] + pixels[i+2]*pixels[i+2];
        }}
        const mean = sum / n;
        const variance = (sumSq / n) - (mean * mean);

        return {{
          meshCount,
          variance,
          bodyPos: [bodies[1].position.x, bodies[1].position.y, bodies[1].position.z],
          bodyQuat: [bodies[1].quaternion.x, bodies[1].quaternion.y, bodies[1].quaternion.z, bodies[1].quaternion.w]
        }};
      }};

      window.ready = true;
    }}

    init();
  </script>
</body>
</html>
"""
        html_file = tmp_path / "index.html"
        html_file.write_text(html_content, encoding="utf-8")

        # Start ephemeral HTTP server
        handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
            *args, directory=str(tmp_path), **kwargs
        )
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
            port = httpd.server_address[1]
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()

            print(f"[ViewerSmokeTest] Running Playwright against http://127.0.0.1:{port}/index.html...")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(executable_path=chrome_path, headless=True)
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
                page.wait_for_function("window.ready === true", timeout=10000)

                report = page.evaluate("window.getReport()")
                browser.close()

            print(f"[ViewerSmokeTest] Report received: {report}")

            # 1. Canvas not blank
            variance = float(report["variance"])
            assert (
                variance > MIN_PIXEL_VARIANCE
            ), f"Canvas appears blank (pixel variance {variance:.2f} <= {MIN_PIXEL_VARIANCE})"

            # 2. Mesh count equals visual geom count
            assert report["meshCount"] == expected_mesh_count, (
                f"Rendered mesh count {report['meshCount']} != expected {expected_mesh_count}"
            )

            # 3. Known body's world pose round-trips
            pos = report["bodyPos"]
            assert np.allclose(pos, [0.1, 0.2, 0.3], atol=1e-3), f"Body position mismatch: {pos}"

            # xyzw quaternion check
            quat = report["bodyQuat"]
            expected_quat = [qx, 0.0, 0.0, qw]
            assert np.allclose(quat, expected_quat, atol=1e-3), (
                f"Body quaternion mismatch: {quat} != {expected_quat}"
            )

            print(
                f"[ViewerSmokeTest] PASS: Canvas variance {variance:.1f} > {MIN_PIXEL_VARIANCE}, "
                f"Mesh count {report['meshCount']} == {expected_mesh_count}, "
                f"Pose round-tripped with non-trivial quat."
            )


if __name__ == "__main__":
    # Ensure export_meshes is importable from same directory
    sys.path.insert(0, str(Path(__file__).parent))
    run_viewer_smoke_test()
