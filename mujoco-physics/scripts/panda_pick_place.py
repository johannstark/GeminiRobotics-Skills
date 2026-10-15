#!/usr/bin/env python3
"""Tested end-to-end MuJoCo reference: Franka Panda pick-and-place with camera deprojection.

What it does (headless, no API key needed):
  1. Builds a scene with MjSpec: Menagerie Panda + TCP site + red cube + gray tray + top camera.
  2. Renders RGB / depth / segmentation from the top camera.
  3. Converts a [y, x] 0-1000 point (the Gemini Robotics-ER format) to a 3D world point.
     By default it fakes the ER point by projecting the cube's true position, so you can
     check the round trip. Pass --er-point Y X to use a real model output instead.
  4. Solves damped-least-squares IK and runs a waypoint pick-and-place.
  5. Verifies the grasp and the placement with contacts, and optionally records an MP4
     that you can send to ER for moment finding / progress classification.

Usage:
  pip install mujoco numpy pillow "imageio[ffmpeg]"
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
  python panda_pick_place.py --menagerie ./mujoco_menagerie --video episode.mp4

  Linux headless: MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU).
  Exit code 0 = cube ended up in the tray.

Import it as a library too: build_scene, render_rgbd, er_point_to_world, world_to_er_point,
solve_ik, pick_and_place, touching.
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, 180 deg about X: tool z-axis points at the table
OPEN, CLOSED = 255.0, 0.0             # Menagerie Panda actuator8 ctrlrange is [0, 255]
CUBE_START = np.array([0.45, 0.3, 0.02])
TRAY_POS = np.array([0.45, -0.3, 0.0])


# ---------------------------------------------------------------- scene

def build_scene(menagerie: str, width: int = 640, height: int = 480):
    """Panda + objects + fixed perception camera. Returns (model, data)."""
    spec = mujoco.MjSpec.from_file(os.path.join(menagerie, "franka_emika_panda", "scene.xml"))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034])  # between the fingertips

    cube = spec.worldbody.add_body(name="cube", pos=CUBE_START.tolist())
    cube.add_freejoint()
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02],
                  rgba=[0.9, 0.1, 0.1, 1], mass=0.05, friction=[2, 0.3, 0.1], condim=4)

    tray = spec.worldbody.add_body(name="tray", pos=TRAY_POS.tolist())
    tray.add_geom(name="tray_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.005],
                  rgba=[0.2, 0.2, 0.2, 1])

    # Identity orientation = looking straight down -Z (image up = world +Y). In the Menagerie "home"
    # pose the arm sits over x~0.3-0.55, y~0, so objects are placed left/right of it to stay visible.
    spec.worldbody.add_camera(name="top", pos=[0.45, 0.0, 1.4], fovy=60)

    # Menagerie keyframes only list the robot's qpos: append x y z qw qx qy qz for the new free joint.
    for key in spec.keys:
        key.qpos = list(key.qpos) + CUBE_START.tolist() + [1, 0, 0, 0]

    spec.visual.global_.offwidth = max(width, 640)   # offscreen buffer must be >= render size
    spec.visual.global_.offheight = max(height, 480)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


# ---------------------------------------------------------------- cameras

def render_rgbd(renderer: mujoco.Renderer, data, camera: str):
    """Returns (rgb uint8 HxWx3, depth float32 HxW in metres, seg int32 HxWx2 [obj_id, obj_type])."""
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return rgb, depth, seg


def intrinsics(model, cam_id: int, width: int, height: int):
    """MuJoCo cameras: vertical fovy, square pixels, principal point at the image centre."""
    f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    return f, f, width / 2, height / 2


def pixel_to_world(model, data, camera: str, u: float, v: float, depth_img):
    """Pixel (u right, v down) + metric depth -> world point. Median over a 5x5 patch."""
    cam_id = model.camera(camera).id
    h, w = depth_img.shape
    fx, fy, cx, cy = intrinsics(model, cam_id, w, h)
    ui, vi = int(np.clip(round(u), 0, w - 1)), int(np.clip(round(v), 0, h - 1))
    d = float(np.median(depth_img[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3]))
    p_cam = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])  # MuJoCo cam: x right, y up, looks along -z
    return data.cam_xpos[cam_id] + data.cam_xmat[cam_id].reshape(3, 3) @ p_cam


def world_to_pixel(model, data, camera: str, p, width: int, height: int):
    cam_id = model.camera(camera).id
    fx, fy, cx, cy = intrinsics(model, cam_id, width, height)
    pc = data.cam_xmat[cam_id].reshape(3, 3).T @ (np.asarray(p) - data.cam_xpos[cam_id])
    return cx + fx * pc[0] / -pc[2], cy - fy * pc[1] / -pc[2]


def er_point_to_world(model, data, camera: str, yx, depth_img):
    """Gemini Robotics-ER point [y, x] normalised 0-1000 -> world xyz (top surface of what was hit)."""
    h, w = depth_img.shape
    return pixel_to_world(model, data, camera, yx[1] / 1000 * w, yx[0] / 1000 * h, depth_img)


def world_to_er_point(model, data, camera: str, p, width: int, height: int):
    u, v = world_to_pixel(model, data, camera, p, width, height)
    return [int(round(v / height * 1000)), int(round(u / width * 1000))]


def pixel_is_geom(model, seg, yx, geom_name: str) -> bool:
    """Occlusion check: does the ER point land on the expected geom in the segmentation image?"""
    h, w = seg.shape[:2]
    v, u = int(yx[0] / 1000 * (h - 1)), int(yx[1] / 1000 * (w - 1))
    obj_id, obj_type = seg[v, u]
    return obj_type == mujoco.mjtObj.mjOBJ_GEOM and obj_id == model.geom(geom_name).id


# ---------------------------------------------------------------- IK + control

def solve_ik(model, data, site: str, target_pos, target_quat=DOWN, iters=300, damping=1e-2,
             tol=1e-4, step=0.5, joints=ARM_JOINTS):
    """Damped least squares on a scratch MjData. Returns arm joint angles or None."""
    ik = mujoco.MjData(model)
    ik.qpos[:] = data.qpos                                         # seed from the current pose
    sid = model.site(site).id
    dofs = np.array([model.joint(j).dofadr[0] for j in joints])
    qadr = np.array([model.joint(j).qposadr[0] for j in joints])
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    q_site, q_conj, q_err, rot_err = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
    for _ in range(iters):
        mujoco.mj_forward(model, ik)
        pos_err = np.asarray(target_pos) - ik.site_xpos[sid]
        mujoco.mju_mat2Quat(q_site, ik.site_xmat[sid])
        mujoco.mju_negQuat(q_conj, q_site)
        mujoco.mju_mulQuat(q_err, target_quat, q_conj)
        mujoco.mju_quat2Vel(rot_err, q_err, 1.0)
        err = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(err) < tol:
            return ik.qpos[qadr].copy()
        mujoco.mj_jacSite(model, ik, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, dofs]
        dq = np.zeros(model.nv)
        dq[dofs] = step * (J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err))
        mujoco.mj_integratePos(model, ik.qpos, dq, 1.0)            # correct for any joint type
        for j in joints:
            jid = model.joint(j).id
            if model.jnt_limited[jid]:
                a = model.jnt_qposadr[jid]
                ik.qpos[a] = np.clip(ik.qpos[a], *model.jnt_range[jid])
    return None


def move_to(model, data, q_target, gripper: float, duration: float, record=None, fps: int = 10):
    """Smoothstep joint interpolation on position actuators ctrl[0:7]; gripper on ctrl[7]."""
    q_start = data.ctrl[:7].copy()
    steps = max(1, int(duration / model.opt.timestep))
    every = max(1, int(1 / (fps * model.opt.timestep)))
    for k in range(steps):
        s = (k + 1) / steps
        s = s * s * (3 - 2 * s)
        data.ctrl[:7] = q_start + (q_target - q_start) * s
        data.ctrl[7] = gripper
        mujoco.mj_step(model, data)
        if record is not None and k % every == 0:
            record(data)


def touching(model, data, body_a: str, body_b: str) -> bool:
    a, b = model.body(body_a).id, model.body(body_b).id
    for i in range(data.ncon):
        c = data.contact[i]
        if {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} == {a, b}:
            return True
    return False


def pick_and_place(model, data, pick, place, record=None, hover=0.15):
    """Waypoints: hover, descend, close, lift (+grasp check), transit, lower, release, retreat."""
    pick, place = np.asarray(pick, float), np.asarray(place, float)
    up = np.array([0, 0, hover])
    plan = [
        ("hover", pick + up, OPEN, 2.0),
        ("descend", pick, OPEN, 1.5),
        ("close", pick, CLOSED, 1.0),
        ("lift", pick + up, CLOSED, 1.5),
        ("transit", place + up, CLOSED, 2.0),
        ("lower", place + [0, 0, 0.05], CLOSED, 1.0),
        ("release", place + [0, 0, 0.05], OPEN, 0.8),
        ("retreat", place + up, OPEN, 1.0),
    ]
    for name, pos, grip, dur in plan:
        q = solve_ik(model, data, "tcp", pos)
        if q is None:
            return False, f"IK failed at '{name}' {np.round(pos, 3).tolist()}"
        move_to(model, data, q, grip, dur, record)
        if name == "lift" and not (touching(model, data, "left_finger", "cube")
                                   or touching(model, data, "right_finger", "cube")):
            return False, "grasp lost after lift"
    return True, "ok"


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--menagerie", required=True, help="path to a mujoco_menagerie checkout")
    ap.add_argument("--er-point", nargs=2, type=int, metavar=("Y", "X"),
                    help="ER point for the cube, [y, x] in 0-1000 (default: project the true cube position)")
    ap.add_argument("--video", help="write an MP4 of the episode (needs imageio[ffmpeg])")
    ap.add_argument("--snapshot", help="write the top-camera RGB used for perception as PNG")
    args = ap.parse_args(argv)

    W, H = 640, 480
    model, data = build_scene(args.menagerie, W, H)
    renderer = mujoco.Renderer(model, height=H, width=W)
    rgb, depth, seg = render_rgbd(renderer, data, "top")
    if args.snapshot:
        from PIL import Image
        Image.fromarray(rgb).save(args.snapshot)

    yx = args.er_point or world_to_er_point(model, data, "top", data.body("cube").xpos, W, H)
    if not pixel_is_geom(model, seg, yx, "cube_geom"):
        print(f"warning: ER point {yx} does not land on the cube (occluded or wrong)", file=sys.stderr)
    surface = er_point_to_world(model, data, "top", yx, depth)
    pick = surface - [0, 0, 0.02]            # ray hits the top face: grasp at the cube centre
    place = data.body("tray").xpos + [0, 0, 0.025]
    print(f"ER point {yx} -> surface {np.round(surface, 4).tolist()} "
          f"(true cube {np.round(data.body('cube').xpos, 4).tolist()})")

    frames = []
    rec = None
    if args.video:
        rec_r = mujoco.Renderer(model, height=240, width=320)

        def rec(d):
            views = []
            for cam in ("top", -1):          # -1 = default free camera; grid views like the ER notebook
                rec_r.update_scene(d, camera=cam)
                views.append(rec_r.render().copy())
            frames.append(np.hstack(views))

    ok, msg = pick_and_place(model, data, pick, place, rec)
    for _ in range(int(0.5 / model.opt.timestep)):   # let the cube settle
        mujoco.mj_step(model, data)
    in_tray = touching(model, data, "cube", "tray")
    print(f"pick_and_place: {ok} ({msg}); cube touching tray: {in_tray}; "
          f"cube at {np.round(data.body('cube').xpos, 3).tolist()}")

    if args.video and frames:
        import imageio.v3 as iio
        iio.imwrite(args.video, np.stack(frames), fps=10, codec="libx264")
        print(f"wrote {args.video} ({len(frames)} frames, views left-to-right: top, free)")
    return 0 if (ok and in_tray) else 1


if __name__ == "__main__":
    sys.exit(main())
