# Real RGB camera grounding with an ArUco mat

## Contents
- [1. Mat design and coordinate frames](#1-mat-design-and-coordinate-frames)
- [2. Image-to-table homography](#2-image-to-table-homography)
- [3. Camera pose from nominal intrinsics](#3-camera-pose-from-nominal-intrinsics)
- [4. Focus lock and half-pixel alignment](#4-focus-lock-and-half-pixel-alignment)
- [5. Parallax correction for elevated targets](#5-parallax-correction-for-elevated-targets)
- [6. Object footprint vs top face](#6-object-footprint-vs-top-face)
- [7. Symptom to fix table](#7-symptom-to-fix-table)
- [8. Mat and camera setup checklist](#8-mat-and-camera-setup-checklist)

---

## 1. Mat design and coordinate frames

An RGB-only webcam cannot measure depth directly. Grounding relies on an ArUco fiducial mat placed on
the table. Four markers (OpenCV dictionary `cv2.aruco.DICT_4X4_50`, IDs 0 to 3) define the boundary
of the manipulation workspace.

```
 Marker 0 (top-left)        Marker 1 (top-right)
       +----------------------------+
       |                            |  Table plane (z = 0)
       |       Robot Workspace      |
       |                            |
       +----------------------------+
 Marker 3 (bottom-left)     Marker 2 (bottom-right)
               [Robot Base]
```

### The IK/FK frame origin trap
The mat corner coordinates in millimeters **must match the robot's IK/FK base frame**, not an external tape
measure:
- On arms such as the SO-101, the kinematic root origin is situated **38.8 mm behind the pan axis**
  (the `shoulder` body's offset `pos` inside `base` in the MJCF model).
- Measuring mat corners with physical tape from the visible pan axis creates a constant **35–45 mm X-offset miss**
  on every commanded reach.

### Teach-by-touch calibration
Instead of ruler measurements, teach the marker corner coordinates directly:
1. Jog the arm manually until the gripper tip or alignment pin touches the center of Marker 0.
2. Record forward kinematics `FK(q)[:2]`.
3. Repeat for Markers 1, 2, and 3.

Touching corners absorbs both the base frame translation and any residual rotation (~4° from motor zero
offsets or mat tape skew).

## 2. Image-to-table homography

A planar homography $H$ maps pixel coordinates $[u, v, 1]^T$ to metric table coordinates $[X, Y, 1]^T$ on
the plane $Z = 0$.

```python
import cv2
import numpy as np

# corners_px: Nx2 detected ArUco centers; corners_table: Nx2 metric coords in IK frame
H, _ = cv2.findHomography(corners_px, corners_table, method=0)
```

### Always use the sent image dimensions
ER returns normalized points $[y, x]$ on a $0\text{–}1000$ grid relative to the image passed to it.
Convert to pixels using the **actual width and height of the image sent in that request**:

```python
u = (x / 1000.0) * sent_frame_width    # never hardcode 640 or 480
v = (y / 1000.0) * sent_frame_height
table_h = H @ np.array([u, v, 1.0])
x_table_mm, y_table_mm = table_h[0] / table_h[2], table_h[1] / table_h[2]
```

Hardcoding image resolution (such as assuming $640\times 480$) when capturing at $1280\times 720$ causes
systematic coordinate scaling errors.

## 3. Camera pose from nominal intrinsics

To localize objects elevated above the table plane ($h > 0$), recover the 3D camera position $C = [C_x, C_y, C_z]$
in table frame coordinates using `cv2.solvePnPGeneric` with `SOLVEPNP_IPPE`.

### Do not fit focal length from a single plane
Estimating camera focal length from a single planar homography is mathematically ill-conditioned: a mere
$0.5\text{ px}$ detection noise can throw the recovered camera position off by up to $1\text{ meter}$.

Instead, construct nominal pinhole intrinsics $K$ using the manufacturer's active horizontal field of
view (HFOV) for the current sensor mode:

```python
fx = (width / 2.0) / np.tan(np.radians(nominal_hfov_deg) / 2.0)
fy = fx
K = np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]], dtype=np.float64)

# Run IPPE on table plane 3D points [X, Y, 0] and corresponding image pixels [u, v]
_, rvecs, tvecs, errs = cv2.solvePnPGeneric(pts_3d_table, pts_img, K, None, flags=cv2.SOLVEPNP_IPPE)

# Pick the physical solution where the camera is positioned above the table (Cz > 0)
for rvec, tvec in zip(rvecs, tvecs):
    R_cw, _ = cv2.Rodrigues(rvec)
    C = -R_cw.T @ tvec.flatten()
    if C[2] > 0:  # Camera is above the workspace
        break
```

### The HFOV error budget
Using the wrong nominal HFOV severely degrades camera height recovery:
- In an SO-101 app with a $356\text{ mm}$-high webcam, assuming a generic $70^\circ$ HFOV instead of the true
  $62.2^\circ$ HFOV collapsed recovered camera height to $297\text{ mm}$ (a $59\text{ mm}$ error).
- At gripper hover height ($h = 100\text{ mm}$), that $59\text{ mm}$ height miss introduces **$35\text{ to }120\text{ mm}$
  of horizontal parallax error**.

## 4. Focus lock and half-pixel alignment

1. **Lock camera auto-focus**: When a webcam auto-focuses, its effective focal length changes dynamically,
   invalidating the homography and intrinsics $K$. Set fixed focus before capturing:
   ```bash
   v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0 -c focus_absolute=20
   ```
   Or in OpenCV: `cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)`.
2. **Half-pixel coordinate convention**: OpenCV ArUco detector corners represent discrete pixel coordinates
   (top-left origin at `(0, 0)`). When feeding continuous points between MuJoCo projections and OpenCV detections,
   add $+0.5\text{ px}$ to match continuous sensor ray centers.

## 5. Parallax correction for elevated targets

The homography $H$ maps image pixels down to the table surface $Z = 0$. If an object or fiducial tag sits at
elevation $h > 0$ above the table (such as a tag mounted on the gripper jaws), its naive homography projection
is shifted away from the camera along the optical ray.

```
       Camera C [Cx, Cy, Cz]
          \
           \
            * Elevated feature [x, y, h]
             \
              \
   ------------+-------------------- Table plane z = 0
             xy_naive = H(u, v)
```

Correct this displacement using the **deparallax formula**:

$$\mathbf{xy}_{\text{true}} = \mathbf{C}_{xy} + (\mathbf{xy}_{\text{table}} - \mathbf{C}_{xy}) \cdot \frac{C_z - h}{C_z}$$

### Worked numerical example
- Camera center recovered from IPPE: $\mathbf{C} = [0.0, -350.0, 356.0]\text{ mm}$ ($C_z = 356\text{ mm}$).
- Gripper fiducial tag height: $h = 100.0\text{ mm}$.
- True physical tag position: $[480.0, 250.0, 100.0]\text{ mm}$.
- Naive table hit through $H$: $\mathbf{xy}_{\text{table}} = [667.4, 490.6]\text{ mm}$ (an error of over $200\text{ mm}$!).
- Applying deparallax:
  $$\text{scale} = \frac{356.0 - 100.0}{356.0} = 0.7191$$
  $$x_{\text{true}} = 0.0 + (667.4 - 0.0) \times 0.7191 = 480.0\text{ mm}$$
  $$y_{\text{true}} = -350.0 + (490.6 - (-350.0)) \times 0.7191 = 250.0\text{ mm}$$

Residual error drops from $>200\text{ mm}$ down to $<1\text{ mm}$.

> [!WARNING]
> Never subtract image points that sit at different heights (e.g. subtracting an object center from a gripper tag).
> Unproject each point to 3D with its known height first, then subtract in metric table space.

## 6. Object footprint vs top face

When ER points to a 3D object on the table:
- **Top face pointing**: The top surface of a $30\text{ mm}$ cube sits at $h = 30\text{ mm}$. Directly unprojecting
  the top face through $H$ shifts the estimated footprint horizontally due to parallax.
- **Bottom contact footprint**: Either prompt ER to identify the bottom contact point with the table, or use
  a 2D segmentation/blob detector on the sent image and take the lowest boundary point through $H$.
- For known object height $h_{\text{obj}}$, unprojecting the visual center requires deparallax with $h = h_{\text{obj}} / 2$.

## 7. Symptom to fix table

| Symptom | Probable cause | Corrective action |
|---|---|---|
| Constant $35\text{–}45\text{ mm}$ X-offset across entire workspace | Corner coordinates measured from visible pan axis instead of kinematic origin | Align table coordinates with the arm's base kinematic root ($38.8\text{ mm}$ behind shoulder pan on SO-101). |
| Positional miss grows proportionally with X or Y | Skewed mat or uncalibrated motor zero offsets causing rotational tilt | Calibrate mat corners by touching them with the arm (teach-by-touch). |
| Gripper tag or object localized with 20–100 mm error along camera ray | Parallax displacement or incorrect assumed sensor HFOV | Apply `deparallax(C, xy, h)` using camera pose recovered with true nominal HFOV. |
| Camera height recovered as $<300\text{ mm}$ when physically mounted at $>350\text{ mm}$ | Wrong nominal HFOV supplied to `camera_from_homography` | Look up sensor mode datasheet (e.g., $62.2^\circ$ vs $70^\circ$) or verify crop settings. |
| Unprojection shifts unpredictably across frames | Dynamic camera autofocus shifting focal length $K$ | Lock camera focus permanently via `CAP_PROP_AUTOFOCUS=0` and fixed focus value. |

## 8. Mat and camera setup checklist

Use this copyable verification feedback loop whenever setting up a new webcam workspace:

- [ ] **1. Print and secure workspace mat**: 4 ArUco markers (`DICT_4X4_50`, IDs 0 to 3) taped flat to table.
- [ ] **2. Teach corner coordinates in IK frame**:
      Jog arm to touch each marker center. Record `[x, y]` from FK. Ensure coordinate frame origin matches kinematic root.
- [ ] **3. Lock webcam parameters**:
      Disable auto-focus and auto-exposure. Verify frame dimensions match downstream processing.
- [ ] **4. Measure/lookup sensor HFOV**:
      Record active horizontal FOV for sensor aspect ratio (e.g. $62.2^\circ$ for standard 16:9 mode).
- [ ] **5. Run geometric selftest**:
      Execute `python scripts/table_grounding.py --selftest` to ensure math and OpenCV installation are sound.
- [ ] **6. Verify reachable workspace**:
      Check that all 4 mat corners and intermediate pickup grid points fall within top-down reach envelope ($\sim 390\text{ mm}$ radius for SO-101).
      *If any corner exceeds envelope, reposition mat closer to base and restart at Step 2.*
