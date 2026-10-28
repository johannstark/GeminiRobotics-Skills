#!/usr/bin/env python3
"""Table-plane homography, camera pose recovery, and deparallax geometry.

Uses OpenCV and NumPy only. Grounding pipeline for an overhead/oblique webcam looking
at an ArUco fiducial mat on a table.

Functions:
  homography_from_markers()            Compute pixel -> table mm homography matrix H
  yx_to_table_mm(H, yx, W, H_img)       Map normalized [y, x] 0-1000 to table mm (z=0)
  camera_from_homography(H, hfov, W, H) Recover camera center C and rotation R via IPPE
  deparallax(C, xy_on_table, h_mm)     Correct table-plane projection for elevation h

Run selftest:
    python scripts/table_grounding.py --selftest
"""

from __future__ import annotations

import argparse
import sys
import cv2
import numpy as np


def homography_from_markers(
    corners_px: np.ndarray | list,
    corners_table_mm: np.ndarray | list,
) -> np.ndarray:
    """Compute 3x3 homography H mapping image pixels [u, v, 1]^T to table mm [X, Y, 1]^T.

    Args:
        corners_px: Nx2 array of pixel coordinates (at least 4 points).
        corners_table_mm: Nx2 array of corresponding table coordinates in mm (IK base frame).

    Returns:
        3x3 float64 homography matrix H such as X_table ~ H @ [u, v, 1]^T.
    """
    px = np.asarray(corners_px, dtype=np.float64)
    tab = np.asarray(corners_table_mm, dtype=np.float64)

    # 4 points is the minimum required to solve 8 degrees of freedom in planar homography
    if len(px) < 4 or len(tab) < 4:
        raise ValueError(
            f"Expected at least 4 marker corners, got {len(px)} image points and "
            f"{len(tab)} table points. Check ArUco marker detection."
        )
    if px.shape != tab.shape:
        raise ValueError(
            f"Shape mismatch: corners_px {px.shape} vs corners_table_mm {tab.shape}."
        )

    # cv2.findHomography maps src (pixels) to dst (table mm)
    H, status = cv2.findHomography(px, tab, method=0)
    if H is None or np.isnan(H).any():
        raise ValueError(
            "Degenerate homography: cv2.findHomography failed. Ensure marker corners "
            "are non-collinear and ordered identically in image and table coordinates."
        )

    # Determinant check: a near-singular matrix indicates degenerate geometry
    if abs(np.linalg.det(H)) < 1e-12:
        raise ValueError(
            "Degenerate homography: determinant is near zero. Verify corner coordinates "
            "are not collinear or improperly scaled."
        )

    return H.astype(np.float64)


def yx_to_table_mm(
    H: np.ndarray,
    yx: list[float] | tuple[float, float] | np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Map ER normalized [y, x] in range 0-1000 to metric table coordinates (X, Y) in mm.

    Uses the actual resolution of the image sent to the model (never hardcode 640x480).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got width={width}, height={height}.")

    # Normalized 0-1000 coordinates to pixel coordinates (u = x, v = y)
    u = float(yx[1]) / 1000.0 * width
    v = float(yx[0]) / 1000.0 * height

    pt_h = H @ np.array([u, v, 1.0], dtype=np.float64)
    # Check for point at infinity (projective horizon)
    if abs(pt_h[2]) < 1e-9:
        raise ValueError(f"Projective coordinate at infinity for image point [{u}, {v}].")

    x_mm = pt_h[0] / pt_h[2]
    y_mm = pt_h[1] / pt_h[2]
    return float(x_mm), float(y_mm)


def camera_from_homography(
    H: np.ndarray,
    hfov_deg: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover camera optical center C (3,) and rotation R (3,3) in table frame.

    Uses nominal intrinsics derived from the true horizontal FOV and IPPE (Infinitesimal
    Plane-based Pose Estimation). Do NOT estimate focal length from a single-plane
    homography, as it is ill-conditioned (0.5 px noise can cause up to 1 m camera center error).

    Returns:
        C: (3,) camera center [X, Y, Z] in table frame (mm).
        R: (3,3) camera orientation matrix in table frame (columns are camera X, Y, Z axes).
    """
    if hfov_deg <= 0 or hfov_deg >= 180:
        raise ValueError(f"hfov_deg must be between 0 and 180 degrees, got {hfov_deg}.")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame dimensions: width={width}, height={height}.")

    # Nominal pinhole intrinsic matrix K assuming zero skew and principal point at center
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    fy = fx  # Square pixels assumption for standard webcams
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Generate a distributed 4-point grid across the image to evaluate PnP
    # Inset by 10% of frame boundary to stay within reliable lens area
    u_lo, u_hi = 0.1 * width, 0.9 * width
    v_lo, v_hi = 0.1 * height, 0.9 * height
    img_pts = np.array([
        [u_lo, v_lo],
        [u_hi, v_lo],
        [u_hi, v_hi],
        [u_lo, v_hi],
    ], dtype=np.float64)

    # Unproject image points through homography to table plane (z=0)
    img_h = np.column_stack([img_pts, np.ones(4, dtype=np.float64)])
    tab_h = (H @ img_h.T).T
    tab_xy = tab_h[:, :2] / tab_h[:, 2:3]
    obj_3d = np.column_stack([tab_xy, np.zeros(4, dtype=np.float64)])

    # Solve PnP using IPPE for planar targets (returns up to 2 solutions)
    success, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
        obj_3d, img_pts, K, distCoeffs=None, flags=cv2.SOLVEPNP_IPPE
    )
    if not success or len(rvecs) == 0:
        raise ValueError("solvePnPGeneric failed to recover camera pose from planar homography.")

    # Find candidate solutions with camera above the table (Cz > 0)
    valid_candidates = []
    for rvec, tvec, err in zip(rvecs, tvecs, reproj_errs):
        R_cw, _ = cv2.Rodrigues(rvec)  # World-to-camera rotation
        R_wc = R_cw.T                   # Camera-to-world rotation
        C = -R_wc @ tvec.flatten()      # Camera center in world coordinates
        # Camera must be physically positioned above the table plane
        if C[2] > 0:
            valid_candidates.append((err[0], C, R_wc))

    if not valid_candidates:
        raise ValueError(
            "All recovered camera positions have Cz <= 0 (camera below table). "
            "Verify table coordinate frame handedness and marker order."
        )

    # Select the physically valid solution that minimizes reprojection error
    valid_candidates.sort(key=lambda item: item[0])
    best_err, best_C, best_R = valid_candidates[0]

    return best_C, best_R


def deparallax(
    C: np.ndarray | list[float],
    xy_on_table: np.ndarray | list[float] | tuple[float, float],
    h_mm: float,
) -> np.ndarray:
    """Correct table-plane projection (z=0) for a feature elevated at height h_mm.

    Formula:
        xy = C_xy + (xy_table - C_xy) * (C_z - h) / C_z

    Args:
        C: 3D camera position [Cx, Cy, Cz] in mm.
        xy_on_table: 2D coordinates [X, Y] obtained from unprojecting to z=0 via H.
        h_mm: Elevation of the feature above table plane in mm (e.g. gripper fiducial height).

    Returns:
        Corrected 2D position [X, Y] in mm in the table plane.
    """
    C_vec = np.asarray(C, dtype=np.float64).flatten()
    if len(C_vec) < 3:
        raise ValueError(f"Camera center C must have 3 coordinates [X, Y, Z], got {C_vec}.")

    Cz = C_vec[2]
    # Camera must be higher than the target point to form a valid ray intersection
    if Cz <= h_mm:
        raise ValueError(
            f"Camera height Cz ({Cz:.1f} mm) must be strictly greater than target elevation h_mm ({h_mm:.1f} mm)."
        )

    xy_tab = np.asarray(xy_on_table, dtype=np.float64)[:2]
    C_xy = C_vec[:2]

    # Geometric ray-plane intersection scaling along the line of sight
    scale = (Cz - h_mm) / Cz
    xy_true = C_xy + (xy_tab - C_xy) * scale
    return xy_true


def _selftest() -> None:
    """Run verification selftest for table grounding, camera recovery, and deparallax."""
    # Frame size typical of USB RGB webcam streaming
    W, H_img = 1280, 720
    # True active horizontal FOV of sensor mode in SO-101 setup
    hfov_true = 62.2
    fx_true = (W / 2.0) / np.tan(np.radians(hfov_true) / 2.0)
    K_true = np.array([[fx_true, 0.0, W / 2.0], [0.0, fx_true, H_img / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Camera center: mounted oblique in front of table at 356 mm height
    C_true = np.array([0.0, -350.0, 356.0], dtype=np.float64)
    # Camera looks toward table center [250, 0, 0] mm
    look_at = np.array([250.0, 0.0, 0.0], dtype=np.float64)
    z_axis = (look_at - C_true) / np.linalg.norm(look_at - C_true)
    # Z-up world coordinate convention
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_wc_true = np.column_stack([x_axis, y_axis, z_axis])
    R_cw_true = R_wc_true.T
    t_cw_true = -R_cw_true @ C_true

    # 4 known ArUco mat corners on table plane z=0 (mm)
    corners_table = np.array([
        [100.0, -150.0],
        [450.0, -150.0],
        [450.0, 250.0],
        [100.0, 250.0],
    ], dtype=np.float64)

    # Project table corners into synthetic pinhole camera
    obj_3d = np.column_stack([corners_table, np.zeros(4, dtype=np.float64)])
    pts_cam = (R_cw_true @ obj_3d.T + t_cw_true[:, None]).T
    pts_img = (K_true @ (pts_cam.T / pts_cam[:, 2])).T[:, :2]

    # 1. Compute homography and verify round-trip error < 1 mm
    H = homography_from_markers(pts_img, corners_table)
    for i in range(len(corners_table)):
        # Normalize 0-1000 [y, x] coordinate
        yx = [pts_img[i, 1] / H_img * 1000.0, pts_img[i, 0] / W * 1000.0]
        reprojected_xy = yx_to_table_mm(H, yx, W, H_img)
        err = np.linalg.norm(np.array(reprojected_xy) - corners_table[i])
        assert err < 1.0, f"Round-trip error {err:.3f} mm exceeds 1.0 mm limit at corner {i}"

    # 2. Recover camera pose from H with true HFOV and verify C is within 5 mm
    C_rec, R_rec = camera_from_homography(H, hfov_true, W, H_img)
    camera_err = np.linalg.norm(C_rec - C_true)
    assert camera_err < 5.0, f"Recovered camera position error {camera_err:.3f} mm exceeds 5.0 mm limit"

    # 3. Test deparallax with elevated tag at h = 100 mm (e.g. gripper tag)
    # True physical location of elevated fiducial tag
    P_tag = np.array([480.0, 250.0, 100.0], dtype=np.float64)
    pt_tag_c = R_cw_true @ P_tag + t_cw_true
    tag_px = (K_true @ (pt_tag_c / pt_tag_c[2]))[:2]

    # Naive homography mapping projects directly to table plane z=0
    yx_tag = [tag_px[1] / H_img * 1000.0, tag_px[0] / W * 1000.0]
    naive_tab = np.array(yx_to_table_mm(H, yx_tag, W, H_img))
    naive_err = np.linalg.norm(naive_tab - P_tag[:2])
    assert naive_err > 15.0, f"Naive homography error {naive_err:.1f} mm expected to exceed 15 mm"

    # Deparallax correction using recovered camera
    corrected_xy = deparallax(C_rec, naive_tab, h_mm=100.0)
    depar_err = np.linalg.norm(corrected_xy - P_tag[:2])
    assert depar_err < 1.0, f"Deparallax error {depar_err:.3f} mm exceeds 1.0 mm limit"

    # 4. Assert 70° vs 62.2° HFOV mismatch produces > 30 mm error at h = 100 mm
    hfov_mismatched = 70.0  # Incorrect assumed field of view
    C_mismatched, _ = camera_from_homography(H, hfov_mismatched, W, H_img)
    wrong_corrected = deparallax(C_mismatched, naive_tab, h_mm=100.0)
    mismatch_err = np.linalg.norm(wrong_corrected - P_tag[:2])
    assert mismatch_err > 30.0, (
        f"HFOV mismatch error {mismatch_err:.1f} mm was expected to exceed 30.0 mm"
    )

    # 5. Test error handling for invalid inputs
    try:
        homography_from_markers(pts_img[:3], corners_table[:3])  # < 4 points
        assert False, "Should have raised ValueError for fewer than 4 points"
    except ValueError as e:
        assert "Expected at least 4 marker corners" in str(e)

    try:
        deparallax([0, 0, 50], [100, 100], h_mm=100.0)  # Camera below target elevation
        assert False, "Should have raised ValueError for camera below target elevation"
    except ValueError as e:
        assert "must be strictly greater than target elevation" in str(e)

    print("table_grounding selftest: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table grounding and camera deparallax helpers")
    parser.add_argument("--selftest", action="store_true", help="Run offline geometric selftests")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    else:
        print(__doc__)
