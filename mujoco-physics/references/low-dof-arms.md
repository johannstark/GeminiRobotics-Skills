# Low-DOF robot manipulation: SO-100 / SO-101 / LeRobot arms

## Contents
- [1. Menagerie robotstudio_so101 and STS3215 servos](#1-menagerie-robotstudio_so101-and-sts3215-servos)
- [2. Re-deriving the TCP site](#2-re-deriving-the-tcp-site)
- [3. Friction-based grasp physics with pad bitmasks](#3-friction-based-grasp-physics-with-pad-bitmasks)
- [4. Settle on convergence and gripper stall](#4-settle-on-convergence-and-gripper-stall)
- [5. 5-DOF inverse kinematics and residual refusal](#5-5-dof-inverse-kinematics-and-residual-refusal)
- [6. Reach envelope and workspace layout](#6-reach-envelope-and-workspace-layout)
- [7. Segmented motion planning](#7-segmented-motion-planning)
- [8. Real vs simulation parity](#8-real-vs-simulation-parity)

---

## 1. Menagerie robotstudio_so101 and STS3215 servos

Low-cost arms such as the SO-100 and SO-101 use 5 serial revolute joints plus a single-dof gripper driven
by bus servos (such as Feetech STS3215). In Google DeepMind's MuJoCo Menagerie, the official model is
`robotstudio_so101` (`google-deepmind/mujoco_menagerie/robotstudio_so101`).

```
Base (pan) ──► Shoulder (lift) ──► Upper Arm (elbow) ──► Lower Arm (wrist flex) ──► Wrist (wrist roll) ──► Gripper
   joint 0          joint 1             joint 2                 joint 3                 joint 4          joint 5
```

The STS3215 servos in the Menagerie MJCF are configured with high position gains:
`<position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>`.
In real hardware, these servos are purely proportional ($P$-only), non-backdrivable, and subject to gravity sag.

## 2. Re-deriving the TCP site

In the default URDF and Menagerie MJCF, the site `gripperframe` is located at `[0.012, -0.0002, -0.098]` on
the `gripper` body. This coincides with the inner face of the *fixed* finger.
- **The trap**: Commanding an IK target at an object's visual center will attempt to place the fixed finger
  inside the object, causing collisions during descent or pushing the object away.
- **The fix**: Re-derive the Tool Center Point (TCP) site to sit at the exact geometric midpoint between
  the fixed jaw and the moving jaw when closed on a standard $30\text{ mm}$ object:
  ```python
# Re-derived TCP site centered between jaws on the gripper body
spec.body("gripper").add_site(name="tcp", pos=[0.0125, 0.0, -0.088])
  ```

## 3. Friction-based grasp physics with pad bitmasks

Never use artificial welds (`mj_addEquality`) to fake grasps in research or real-transfer environments.
A realistic grasp must rely purely on physical contact friction:

1. **Drive the arm through `ctrl`, never `qpos`**:
   Teleporting the arm by writing `data.qpos` resets velocities `data.qvel` to zero. A held object
   feels zero acceleration or frictional drag from the moving pads and is immediately left behind.
2. **Thin box friction pads**:
   Replace noisy jaw mesh collision geometries with thin box pads placed on the inner gripping faces:
   ```python
spec.body("gripper").add_geom(
    name="pad_fixed", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.004, 0.015, 0.015], pos=[-0.006, 0.0, -0.088],
    friction=[1.8, 0.005, 0.0001], solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
    contype=2, conaffinity=2, condim=4,
)
   ```
3. **Contact bitmasks (contype and conaffinity)**:
   - Floor and static scenery: `contype=1, conaffinity=1` (bit 0).
   - Friction pads: `contype=2, conaffinity=2` (bit 1). Pads collide with graspable objects (bit 1),
     but cannot collide with the floor (bit 0).
   - Graspable objects (e.g. cube): `contype=3, conaffinity=3` (bits 0 and 1), allowing them to rest on the
     floor and be grasped by pads.
   - Add `<exclude body1="gripper" body2="moving_jaw_so101_v1"/>` so jaws close on air without colliding.
4. **Gripper `forcerange` window**:
   - At default $3.35\text{ N}\cdot\text{m}$, the high squeeze force ejects a lightweight $30\text{ mm}$ cube
     sideways during lift.
   - Below $1.2\text{ N}\cdot\text{m}$, the servo lacks torque to stall firmly before lift begins.
   - Set `forcerange="[-1.5, 1.5]"` on the gripper actuator.

## 4. Settle on convergence and gripper stall

Low-cost servos are lightly damped ($kv \approx 2.73$). A fixed simulation step count is shorter than
one mechanical oscillation.
- **Settle condition**: Step physics until joint position error is $< 0.5^\circ$ ($0.0087\text{ rad}$)
  **and** joint velocities are $< 0.02\text{ rad/s}$, with a $3\text{ s}$ timeout.
- **Gripper stall**: The gripper actuator stalls against the object before reaching its commanded target.
  Treat the gripper as "stopped" when its velocity $|qvel| < 0.02\text{ rad/s}$, not "on target".
- **Aperture verification**:
  When fully closed on air, the jaw reaches $\theta_{\text{empty}} \approx -9.7^\circ$ ($-0.17\text{ rad}$).
  If the stalled aperture is $\le \theta_{\text{empty}} + 6^\circ$, the gripper has closed on air.

## 5. 5-DOF inverse kinematics and residual refusal

A 5-DOF arm cannot satisfy an arbitrary 6-D target pose (such as a full orientation quaternion).
Specifying an over-constrained 6-D pose target will diverge or produce uncontrolled joint spinning.

### Formulation: Position + Tool-Axis-Down
1. **Position (3 DOFs)**: $\mathbf{p}_{\text{tcp}} \to \mathbf{p}_{\text{target}}$.
2. **Tool-axis-down (2 DOFs)**: Tool pointing vector $\mathbf{v}_{\text{tool}} = -R_b[:, 2]$ must align with
   world vertical $-\hat{\mathbf{z}} = [0, 0, -1]$. The angular error is the horizontal cross-product
   $\mathbf{e}_{\text{axis}} = \mathbf{v}_{\text{tool}} \times [0, 0, -1]$.
3. **Wrist roll (yaw)**: Controlled independently by joint 4 (`wrist_roll`) matching the target object's yaw.
4. **Null-space posture bias**: Add a bias toward the neutral seed posture $\mathbf{q}_{\text{seed}} = [0, -0.2, 0.6, 1.2, 0]$
   to keep the elbow high and joints away from limits.

### Fail-closed residual policy
The IK solver must return `(q, residual, mode)`. Caller code must check the residual **before issuing motion**:
```python
q, residual, mode = solve_ik_5dof(model, data, target_pos)
if residual > 0.015:  # 15 mm tolerance
    raise Unreachable(f"Target {target_pos} exceeds 15 mm residual threshold (res={residual*1000:.1f} mm)")
```
- A failed residual check raises a typed error (`Unreachable`, `OrientationUnreachable`) that translates to
  `is_error: True` in the robot tool response.
- **Never fall back silently to position-only IK**: If top-down orientation fails, falling back to a tilted
  grasp without explicit caller consent results in the gripper knocking the object over.
  Position-only fallback is permitted only when explicitly requested (e.g. transit hover).

## 6. Reach envelope and workspace layout

The physical top-down manipulation reach envelope of the SO-101 is approximately **$390\text{ mm}$ radius**
from the base.
- In desktop setups, fiducial mats often extend to $X \approx 450\text{ mm}$. Targets in the $X > 380\text{ mm}$
  strip cannot be reached top-down because `wrist_flex` saturates at $+1.66\text{ rad}$.
- **Pre-flight layout check**: Run this snippet during workspace commissioning to verify all mat corners
  and grid pick locations are reachable:
  ```python
  for pt in workspace_grid_points:
      try:
          solve_ik_5dof(model, data, pt, allow_position_only=False)
      except IKError:
          print(f"WARNING: Workspace point {pt} is unreachable top-down! Reposition mat.")
  ```

## 7. Segmented motion planning

Never interpolate joint angles directly between two low poses on the table. Due to non-linear kinematics,
the gripper dips or sweeps sideways, knocking over adjacent objects.

Always execute Cartesian pick-and-place in **four distinct segments**:
1. **Approach hover**: Move high above pick target ($Z + 6\text{ cm}$).
2. **Vertical descent**: Lower vertically along $Z$ to grasp height with jaws open.
3. **Vertical lift**: After closing gripper and verifying aperture, lift straight up to transit height ($Z + 6\text{ cm}$).
4. **Transit & vertical placement**: Move high to the placement hover point, descend vertically to release height, open jaws, and retract vertically.

## 8. Real vs simulation parity

1. **LeRobot DEGREES mode**: LeRobot configurations often assume motor zero equals the MJCF joint zero.
   Always verify zero calibration with arm extended horizontally forward.
2. **P-only servo gravity sag**: In simulation, the STS3215 actuator has $kp \approx 998$ and holds position
   stiffly. Real hardware sags by $5\text{–}15\text{ mm}$ when extended forward. Add a feed-forward gravity
   compensation term or a contact-guarded descent on real hardware.
3. **Kinematic base offset**: The base origin in MJCF is $38.8\text{ mm}$ behind the shoulder pan axis.
   Real measurements must account for this offset (see the `gemini-robotics-er` skill, `real-camera-grounding.md`).
