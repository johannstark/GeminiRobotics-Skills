#!/usr/bin/env python3
"""Exports compiled MuJoCo meshes and scene geometry for Three.js rendering.

Raw STL meshes stored in robot asset folders often have different coordinate offsets
and origins than what MuJoCo compiles internally (MuJoCo automatically re-centers
and re-orients meshes at compile time). Loading raw STLs in Three.js causes links
to appear misaligned or rotated relative to the physical bodies.

This script exports the compiled geometry directly from MjModel (mesh_vert and mesh_face
using mesh_vertadr and mesh_faceadr), branches strictly on geom_type (never on geom_group),
extracts perception camera parameters, and caches exports keyed by model hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any

import mujoco
import numpy as np

# Manifest schema version for frontend compatibility checks
MANIFEST_VERSION: str = "1.0.0"

# Default relative output directory for web viewer asset caches
DEFAULT_CACHE_DIR: str = "mesh_cache"

# MuJoCo mjtGeom enum mapping to Three.js recognizable geometry names
GEOM_TYPE_NAMES: dict[int, str] = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}

# Candidate perception camera names checked in order of preference
PREFERRED_PERCEPTION_CAM_NAMES: tuple[str, ...] = ("perception", "cam", "overhead", "top")

# Binary mesh file magic bytes to prevent misinterpreting corrupt buffers
MESH_MAGIC: bytes = b"MJMESH01"


def compute_model_hash(model: mujoco.MjModel, source_bytes: bytes | None = None) -> str:
    """Computes a SHA256 digest fingerprinting compiled geometry and mesh vertices."""
    h = hashlib.sha256()
    if source_bytes is not None:
        h.update(source_bytes)
    # Hash the compiled structural counts and topology
    h.update(struct.pack("<IIII", model.nbody, model.ngeom, model.nmesh, model.ncam))
    h.update(model.geom_type.tobytes())
    h.update(model.geom_size.tobytes())
    h.update(model.geom_pos.tobytes())
    h.update(model.geom_quat.tobytes())
    h.update(model.mesh_vert.tobytes())
    h.update(model.mesh_face.tobytes())
    return h.hexdigest()[:16]


def export_model(
    model: mujoco.MjModel,
    out_dir: str | Path,
    source_bytes: bytes | None = None,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Exports compiled MuJoCo meshes and scene manifest into out_dir / <model_hash>.

    Args:
        model: Compiled MuJoCo MjModel.
        out_dir: Base directory for the mesh cache.
        source_bytes: Raw XML bytes if available (used for cache key hashing).
        force: If True, regenerates assets even if cached.

    Returns:
        Tuple of (path_to_manifest_json, manifest_dict).
    """
    model_hash = compute_model_hash(model, source_bytes)
    target_dir = Path(out_dir) / model_hash
    manifest_path = target_dir / "manifest.json"

    if manifest_path.is_file() and not force:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                cached_manifest = json.load(f)
            if cached_manifest.get("model_hash") == model_hash:
                return manifest_path, cached_manifest
        except Exception:
            # Corrupted cache entry; re-export
            pass

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export meshes from compiled mesh_vert / mesh_face buffers
    meshes_manifest: list[dict[str, Any]] = []
    for m_id in range(model.nmesh):
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m_id) or f"mesh_{m_id}"
        v_adr = int(model.mesh_vertadr[m_id])
        v_num = int(model.mesh_vertnum[m_id])
        f_adr = int(model.mesh_faceadr[m_id])
        f_num = int(model.mesh_facenum[m_id])

        # Slices from compiled contiguous buffers
        verts = np.asarray(model.mesh_vert[v_adr : v_adr + v_num], dtype=np.float32)
        faces = np.asarray(model.mesh_face[f_adr : f_adr + f_num], dtype=np.uint32)

        bin_filename = f"mesh_{m_id}_{mesh_name}.bin"
        bin_path = target_dir / bin_filename

        # Write binary mesh: Magic (8B) + nvert (uint32) + nface (uint32) + verts (float32) + faces (uint32)
        with open(bin_path, "wb") as f_bin:
            f_bin.write(MESH_MAGIC)
            f_bin.write(struct.pack("<II", v_num, f_num))
            f_bin.write(verts.tobytes())
            f_bin.write(faces.tobytes())

        meshes_manifest.append(
            {
                "id": m_id,
                "name": mesh_name,
                "nvert": v_num,
                "nface": f_num,
                "bin_file": bin_filename,
                "vert_offset": 16,  # 8 bytes magic + 8 bytes header (nvert, nface)
                "vert_bytes": int(verts.nbytes),
                "face_offset": 16 + int(verts.nbytes),
                "face_bytes": int(faces.nbytes),
            }
        )

    # 2. Export bodies
    bodies_manifest: list[dict[str, Any]] = []
    for b_id in range(model.nbody):
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body_{b_id}"
        parent_id = int(model.body_parentid[b_id])
        bodies_manifest.append({"id": b_id, "name": b_name, "parent_id": parent_id})

    # 3. Export geoms: Branch strictly on geom_type, never on geom_group
    geoms_manifest: list[dict[str, Any]] = []
    for g_id in range(model.ngeom):
        g_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_id) or f"geom_{g_id}"
        g_type = int(model.geom_type[g_id])
        type_name = GEOM_TYPE_NAMES.get(g_type, f"unknown_{g_type}")
        group = int(model.geom_group[g_id])
        body_id = int(model.geom_bodyid[g_id])

        # Half-extents
        size = [float(s) for s in model.geom_size[g_id][:3]]
        pos = [float(p) for p in model.geom_pos[g_id][:3]]
        quat = [float(q) for q in model.geom_quat[g_id][:4]]  # wxyz

        # Color: mat_rgba if matid >= 0, else geom_rgba
        mat_id = int(model.geom_matid[g_id])
        if mat_id >= 0:
            rgba = [float(c) for c in model.mat_rgba[mat_id][:4]]
        else:
            rgba = [float(c) for c in model.geom_rgba[g_id][:4]]

        # Associate mesh_id ONLY if geom_type is mjGEOM_MESH (7)
        mesh_id = int(model.geom_dataid[g_id]) if g_type == 7 else None

        geoms_manifest.append(
            {
                "id": g_id,
                "name": g_name,
                "body_id": body_id,
                "type": g_type,
                "type_name": type_name,
                "size": size,
                "pos": pos,
                "quat": quat,
                "rgba": rgba,
                "group": group,
                "mesh_id": mesh_id,
            }
        )

    # 4. Export perception camera (if defined in the scene)
    perception_cam: dict[str, Any] | None = None
    if model.ncam > 0:
        cam_id: int | None = None
        for pref in PREFERRED_PERCEPTION_CAM_NAMES:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, pref)
            if cid >= 0:
                cam_id = cid
                break
        if cam_id is None:
            cam_id = 0  # Fallback to the first declared camera

        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id) or f"cam_{cam_id}"
        perception_cam = {
            "id": cam_id,
            "name": cam_name,
            "pos": [float(p) for p in model.cam_pos[cam_id][:3]],
            "quat": [float(q) for q in model.cam_quat[cam_id][:4]],  # wxyz
            "fovy": float(model.cam_fovy[cam_id]),
        }

    manifest = {
        "version": MANIFEST_VERSION,
        "model_hash": model_hash,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "bodies": bodies_manifest,
        "geoms": geoms_manifest,
        "meshes": meshes_manifest,
        "perception_camera": perception_cam,
    }

    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)

    return manifest_path, manifest


def export_from_xml(
    xml_path: str | Path,
    out_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Loads a scene XML file and exports compiled assets to out_dir."""
    p = Path(xml_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Scene XML file not found at '{p}'. Provide a valid path to an existing MJCF file."
        )

    try:
        source_bytes = p.read_bytes()
        model = mujoco.MjModel.from_xml_path(str(p))
    except Exception as exc:
        raise ValueError(
            f"Failed to load or compile MuJoCo model from '{p}': {exc}. "
            "Verify XML validity and ensure referenced mesh assets exist in meshdir."
        ) from exc

    return export_model(model, out_dir=out_dir, source_bytes=source_bytes, force=force)


def run_selftest() -> None:
    """Validates exporter functionality, mesh extraction, caching, and error handling."""
    import tempfile

    print("[Selftest] 1. Testing synthetic scene with mesh, box, and perception camera...")
    synthetic_xml = """
    <mujoco model="synthetic_test">
      <asset>
        <mesh name="pyramid" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
      </asset>
      <worldbody>
        <light pos="0 0 2"/>
        <camera name="perception" pos="0 -0.5 0.5" fovy="45"/>
        <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1" group="0"/>
        <body name="arm_link" pos="0 0 0.1">
          <geom name="visual_mesh" type="mesh" mesh="pyramid" rgba="1 0 0 1" group="2"/>
          <geom name="visual_box_in_group2" type="box" size="0.02 0.02 0.02" rgba="0 1 0 1" group="2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(synthetic_xml)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        man_path, man = export_model(model, out_dir=tmp_path, source_bytes=synthetic_xml.encode("utf-8"))

        assert man_path.is_file(), f"Manifest file missing at {man_path}"
        assert man["nmesh"] == 1, f"Expected 1 mesh, got {man['nmesh']}"
        assert man["ngeom"] == 3, f"Expected 3 geoms, got {man['ngeom']}"
        assert man["perception_camera"] is not None, "Missing perception camera in manifest"
        assert man["perception_camera"]["name"] == "perception"

        # Verify geom branching rule: branch on geom_type, never on geom_group
        box_geom = next(g for g in man["geoms"] if g["name"] == "visual_box_in_group2")
        assert box_geom["group"] == 2, "box should be in group 2"
        assert box_geom["type_name"] == "box", f"Expected type 'box', got '{box_geom['type_name']}'"
        assert box_geom["mesh_id"] is None, "Primitive geom must have mesh_id=None even when in group 2"

        mesh_geom = next(g for g in man["geoms"] if g["name"] == "visual_mesh")
        assert mesh_geom["type_name"] == "mesh"
        assert mesh_geom["mesh_id"] == 0

        # Verify binary mesh contents
        mesh_meta = man["meshes"][0]
        bin_path = tmp_path / man["model_hash"] / mesh_meta["bin_file"]
        assert bin_path.is_file(), f"Mesh binary missing at {bin_path}"

        with open(bin_path, "rb") as f_bin:
            magic = f_bin.read(8)
            assert magic == MESH_MAGIC, f"Invalid mesh magic: {magic}"
            nvert, nface = struct.unpack("<II", f_bin.read(8))
            assert nvert == 4, f"Expected 4 vertices, got {nvert}"
            assert nface == 4, f"Expected 4 faces, got {nface}"
            verts_bytes = f_bin.read(nvert * 12)
            faces_bytes = f_bin.read(nface * 12)
            verts = np.frombuffer(verts_bytes, dtype=np.float32).reshape(nvert, 3)
            faces = np.frombuffer(faces_bytes, dtype=np.uint32).reshape(nface, 3)

        assert np.allclose(verts, model.mesh_vert[:4]), "Exported vertices do not match model.mesh_vert"
        assert np.array_equal(faces, model.mesh_face[:4]), "Exported faces do not match model.mesh_face"

        print("[Selftest] 2. Testing cache hit behavior...")
        man_path2, man2 = export_model(model, out_dir=tmp_path, source_bytes=synthetic_xml.encode("utf-8"))
        assert man_path2 == man_path
        assert man2["model_hash"] == man["model_hash"]

        print("[Selftest] 3. Testing actionable error reporting for invalid input...")
        try:
            export_from_xml("/nonexistent/path/to/scene.xml", out_dir=tmp_path)
            assert False, "Failed to raise on missing file"
        except FileNotFoundError as err:
            assert "Scene XML file not found" in str(err)

        try:
            broken_xml = tmp_path / "broken.xml"
            broken_xml.write_text("<invalid xml syntax")
            export_from_xml(broken_xml, out_dir=tmp_path)
            assert False, "Failed to raise on invalid XML"
        except ValueError as err:
            assert "Failed to load or compile" in str(err)

        # 4. Optional Menagerie verification if present
        so101_scene = Path("mujoco_menagerie/robotstudio_so101/scene.xml")
        if so101_scene.is_file():
            print(f"[Selftest] 4. Testing Menagerie scene: {so101_scene}...")
            so101_man_path, so101_man = export_from_xml(so101_scene, out_dir=tmp_path)
            assert so101_man_path.is_file()
            assert so101_man["nmesh"] == 18, f"Expected 18 meshes, got {so101_man['nmesh']}"
            # Check geom 38 (box in group 2) correctly exported without mesh_id
            g38 = so101_man["geoms"][38]
            assert g38["type_name"] == "box", f"geom 38 type should be box, got {g38['type_name']}"
            assert g38["mesh_id"] is None, "geom 38 should have mesh_id None"
            print(f"[Selftest] Menagerie SO-101 verified: 18 meshes, {so101_man['ngeom']} geoms.")

    print("[Selftest] PASS: All exporter assertions and error checks verified.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compiled MuJoCo meshes and scene manifest for Three.js rendering."
    )
    parser.add_argument("xml_path", nargs="?", help="Path to MuJoCo MJCF scene XML")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Target cache directory for web viewer assets (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-export even if cached assets exist",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run comprehensive test suite and exit",
    )

    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        sys.exit(0)

    if not args.xml_path:
        parser.print_help()
        sys.exit(1)

    manifest_path, manifest = export_from_xml(args.xml_path, out_dir=args.out_dir, force=args.force)
    print(f"Exported manifest to {manifest_path} (model_hash: {manifest['model_hash']})")
    print(f"Total: {manifest['nbody']} bodies, {manifest['ngeom']} geoms, {manifest['nmesh']} meshes.")


if __name__ == "__main__":
    main()
