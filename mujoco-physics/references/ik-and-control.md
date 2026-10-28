# Inverse kinematics, control and pick-and-place

Every snippet was executed: Python DLS (`scripts/panda_pick_place.py`), TypeScript DLS (`@mujoco/mujoco` 3.16), and mocap weld.

## Contents
- [Choose a control strategy](#choose-a-control-strategy)
- [Damped least squares IK (Python)](#damped-least-squares-ik-python)
- [5-DOF IK with typed errors and residual checks](#5-dof-ik-with-typed-errors-and-residual-checks)
- [Damped least squares IK (TypeScript / WASM)](#damped-least-squares-ik-typescript--wasm)
- [Mocap + weld (no IK)](#mocap--weld-no-ik)
- [Executing waypoints & settle-on-convergence](#executing-waypoints--settle-on-convergence)
- [Pick-and-place recipe](#pick-and-place-recipe)
- [Grippers](#grippers)
- [The Franka demo's analytical Franka IK](#the-franka-demos-analytical-franka-ik)
- [Debugging IK](#debugging-ik)

## Choose a control strategy

| Strategy | How | Good for | Watch out |
|---|---|---|---|
| **Joint position actuators + IK** (Menagerie default, the Franka demo) | IK → joint targets → interpolate `ctrl` | Realistic arms, sim-to-real parity | IK can fail. Interpolate to avoid jerks. |
| **Mocap + weld** | A mocap body welded to the hand. Write `mocap_pos`/`mocap_quat`. | Fast prototypes, teleop, no IK code | Unrealistic forces. Set `relpose` explicitly. Actuators must not fight it. |
| **Torque / operational space** | `ctrl` from Jacobians, `qfrc_bias` and the mass matrix | Compliant contact, force control | Needs torque actuators and tuning |
| **Learned policy / VLA** | Policy outputs `ctrl` | Dexterous skills | Out of scope here. ER can orchestrate it as a tool. |

## Damped least squares IK (Python)

```python
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
DOWN = np.array([0.0, 1.0, 0.0, 0.0])          # wxyz: 180° about X, tool z points at the table

def solve_ik(model, data, site, target_pos, target_quat=DOWN, iters=300, damping=1e-2,
             tol=1e-4, step=0.5, joints=ARM_JOINTS):
    ik = mujoco.MjData(model)                   # scratch data: never disturbs the running sim
    ik.qpos[:] = data.qpos                      # seed from the current pose (continuity)
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
        mujoco.mj_integratePos(model, ik.qpos, dq, 1.0)        # handles quaternion joints correctly
        for j in joints:
            jid = model.joint(j).id
            if model.jnt_limited[jid]:
                a = model.jnt_qposadr[jid]
                ik.qpos[a] = np.clip(ik.qpos[a], *model.jnt_range[jid])
    return None                                                 # did not converge
```

Variants:

- **Position only:** use `J = jacp[:, dofs]`, `err = pos_err`, and `np.eye(3)`.
- **Weighted orientation:** scale `rot_err` (for example ×0.3) to favour position.
- **Null-space bias towards a rest pose** (7-DOF arms): `dq += (I - J⁺J) · k · (q_rest − q)`.
- **Retry:** if it fails, reseed from a neutral pose (`[0, -0.785, 0, -2.356, 0, 1.571, 0.785]` for the Panda) before
  giving up. An unreachable target returns `None`. **Don't move** when that happens. Report it.

> [!IMPORTANT]
> **Check the residual BEFORE moving.** Never substitute a silent best-effort pose or ignore an unconverged IK return code. If the positional residual exceeds a documented tolerance (e.g., > 15 mm), the caller must raise a typed error (`UnreachableError`) and fail closed. Moving on a partial IK solution causes the arm to swing to an arbitrary configuration and miss or collide.

## 5-DOF IK with typed errors and residual checks

For 5-DOF arms (e.g. SO-100, SO-101, STS3215 class), a 6-DOF Cartesian pose $(x, y, z, roll, pitch, yaw)$ is overconstrained. The standard formulation constrains 3D position plus tool-axis-down alignment (cross product of tool axis with $[0, 0, -1]$ = 0), yielding a 5D error vector.

```python
class IkError(Exception):
    """Base class for inverse kinematics failures."""

class UnreachableError(IkError):
    """Target position lies outside the reachable workspace."""

class OrientationUnreachableError(IkError):
    """Position is reachable but requested tool orientation cannot be satisfied."""

def solve_ik_5dof(model, data, site_name, target_pos,
                  target_axis=np.array([0.0, 0.0, -1.0]),
                  max_residual_mm=15.0, iters=400, damping=1e-2, step=0.5,
                  joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")):
    """Solves 5-DOF IK with position + tool-axis alignment constraints.

    Returns (q, residual_mm, mode).
    Raises UnreachableError if positional residual exceeds max_residual_mm BEFORE motion.
    Raises OrientationUnreachableError if axis alignment cannot be satisfied.
    """
    ik = mujoco.MjData(model)
    ik.qpos[:] = data.qpos
    sid = model.site(site_name).id
    dofs = np.array([model.joint(j).dofadr[0] for j in joints])
    qadr = np.array([model.joint(j).qposadr[0] for j in joints])

    # Pre-seed shoulder pan from target azimuth to avoid local minima
    pan_adr = model.joint(joints[0]).qposadr[0]
    ik.qpos[pan_adr] = np.arctan2(target_pos[1], target_pos[0])

    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_forward(model, ik)
        pos_err = np.asarray(target_pos) - ik.site_xpos[sid]

        # Tool Z-axis in world frame
        tool_z = ik.site_xmat[sid].reshape(3, 3)[:, 2]
        axis_err = np.cross(tool_z, target_axis)

        err5 = np.concatenate([pos_err, axis_err[:2]])
        pos_res_mm = float(np.linalg.norm(pos_err) * 1000.0)
        axis_res = float(np.linalg.norm(axis_err))

        if pos_res_mm < 1.0 and axis_res < 0.05:
            return ik.qpos[qadr].copy(), pos_res_mm, "aligned"

        mujoco.mj_jacSite(model, ik, jacp, jacr, sid)
        J = np.vstack([jacp, jacr[:2]])[:, dofs]
        dq = np.zeros(model.nv)
        dq[dofs] = step * (J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(5), err5))
        mujoco.mj_integratePos(model, ik.qpos, dq, 1.0)
        for j in joints:
            jid = model.joint(j).id
            if model.jnt_limited[jid]:
                a = model.jnt_qposadr[jid]
                ik.qpos[a] = np.clip(ik.qpos[a], *model.jnt_range[jid])

    # Enforce strict residual check before any physical motion
    final_pos_err = float(np.linalg.norm(np.asarray(target_pos) - ik.site_xpos[sid]) * 1000.0)
    if final_pos_err > max_residual_mm:
        raise UnreachableError(
            f"Target position {target_pos} residual {final_pos_err:.1f} mm exceeds tolerance {max_residual_mm} mm"
        )

    tool_z = ik.site_xmat[sid].reshape(3, 3)[:, 2]
    if np.linalg.norm(np.cross(tool_z, target_axis)) > 0.2:
        raise OrientationUnreachableError(
            f"Target {target_pos} reached with {final_pos_err:.1f} mm error but tool axis tilted > 0.2 rad"
        )

    return ik.qpos[qadr].copy(), final_pos_err, "relaxed"
```

## Damped least squares IK (TypeScript / WASM)

This is the same algorithm. It needs `@mujoco/mujoco` ≥ 3.16 (buffers take a size; read them with `GetView()`).

```ts
function solve6(A: number[][], b: number[]): number[] {            // Gaussian elimination, partial pivoting
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}

export function solveIkDls(mujoco: any, m: any, d: any, siteId: number, targetPos: number[], targetQuat: number[],
                           jointIds: number[], { iters = 300, damping = 1e-2, tol = 1e-4, step = 0.5 } = {}): number[] | null {
  const nv = m.nv, ik = new mujoco.MjData(m);
  const jacp = new mujoco.DoubleBuffer(3 * nv), jacr = new mujoco.DoubleBuffer(3 * nv);
  const qSite = new mujoco.DoubleBuffer(4), qConj = new mujoco.DoubleBuffer(4);
  const qErr = new mujoco.DoubleBuffer(4), rotErr = new mujoco.DoubleBuffer(3);
  try {
    ik.qpos.set(d.qpos);                                           // seed from the current pose
    const dofs = jointIds.map(j => m.jnt_dofadr[j]), qadr = jointIds.map(j => m.jnt_qposadr[j]);
    for (let it = 0; it < iters; it++) {
      mujoco.mj_forward(m, ik);
      const p = ik.site_xpos.subarray(3 * siteId, 3 * siteId + 3);
      mujoco.mju_mat2Quat(qSite, Array.from(ik.site_xmat.subarray(9 * siteId, 9 * siteId + 9)));
      mujoco.mju_negQuat(qConj, Array.from(qSite.GetView()));
      mujoco.mju_mulQuat(qErr, targetQuat, Array.from(qConj.GetView()));
      mujoco.mju_quat2Vel(rotErr, Array.from(qErr.GetView()), 1.0);
      const r = rotErr.GetView();
      const err = [targetPos[0] - p[0], targetPos[1] - p[1], targetPos[2] - p[2], r[0], r[1], r[2]];
      if (Math.hypot(...err) < tol) return qadr.map(a => ik.qpos[a]);
      mujoco.mj_jacSite(m, ik, jacp, jacr, siteId);
      const Jp = jacp.GetView(), Jr = jacr.GetView();             // row-major 3 x nv
      const J = [0, 1, 2, 3, 4, 5].map(row => dofs.map(c => row < 3 ? Jp[row * nv + c] : Jr[(row - 3) * nv + c]));
      const A = J.map((ri, i) => J.map((rk, k) => ri.reduce((s, v, c) => s + v * rk[c], 0) + (i === k ? damping ** 2 : 0)));
      const y = solve6(A, err);                                    // (J Jᵀ + λ²I) y = e
      const dq = new Array(nv).fill(0);
      dofs.forEach((dof, c) => { dq[dof] = step * J.reduce((s, row, i) => s + row[c] * y[i], 0); });   // dq = Jᵀ y
      mujoco.mj_integratePos(m, ik.qpos, dq, 1.0);
      for (const j of jointIds) if (m.jnt_limited[j]) {
        const a = m.jnt_qposadr[j];
        ik.qpos[a] = Math.min(Math.max(ik.qpos[a], m.jnt_range[2 * j]), m.jnt_range[2 * j + 1]);
      }
    }
    return null;
  } finally { [jacp, jacr, qSite, qConj, qErr, rotErr, ik].forEach(o => o.delete()); }
}

// usage
const OBJ = mujoco.mjtObj;
const site = mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, 'tcp');
const arm = [1, 2, 3, 4, 5, 6, 7].map(i => mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, `joint${i}`));
const q = solveIkDls(mujoco, model, data, site, [0.45, 0.3, 0.15], [0, 1, 0, 0], arm);
```

For real-time use (60 Hz), keep one scratch `MjData` and the buffers alive between calls instead of
allocating them per solve. With a three.js target, convert with `[q.w, q.x, q.y, q.z]` (MuJoCo is wxyz).

## Mocap + weld (no IK)

```python
target = spec.worldbody.add_body(name="target", mocap=True, pos=[0.5, 0, 0.4], quat=[0, 1, 0, 0])
target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01], contype=0, conaffinity=0, rgba=[0, 1, 0, .5])
eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="target", name2="hand", objtype=mujoco.mjtObj.mjOBJ_BODY)
eq.data[:11] = [0, 0, 0,  0, 0, 0,  1, 0, 0, 0,  1]    # anchor, relpose pos+quat = IDENTITY, torquescale
model = spec.compile(); data = mujoco.MjData(model)
mid = model.body("target").mocapid[0]

data.mocap_pos[mid] = [0.5, 0.2, 0.3]                  # and mocap_quat (wxyz) for orientation
for _ in range(2000):
    data.ctrl[:7] = data.qpos[:7]                      # position actuators FOLLOW instead of resisting the weld
    mujoco.mj_step(model, data)
```

XML equivalent: `<weld body1="target" body2="hand" relpose="0 0 0 1 0 0 0"/>`. In testing, leaving the
MjSpec defaults welded the hand at the wrong offset. The weld keeps whatever relative pose its data encodes,
so always set it explicitly.

## Executing waypoints & settle-on-convergence

Joint-space interpolation with smoothstep easing, on position actuators:

```python
def move_to(model, data, q_target, gripper, duration, record=None, fps=10):
    q_start = data.ctrl[:7].copy()
    steps = max(1, int(duration / model.opt.timestep))
    every = max(1, int(1 / (fps * model.opt.timestep)))
    for k in range(steps):
        s = (k + 1) / steps; s = s * s * (3 - 2 * s)
        data.ctrl[:7] = q_start + (q_target - q_start) * s
        data.ctrl[7] = gripper
        mujoco.mj_step(model, data)
        if record is not None and k % every == 0:
            record(data)

def move_to_settle(model, data, q_target, gripper, max_seconds=3.0,
                   pos_tol=0.0087, vel_tol=0.02, record=None, fps=10):
    """Drives position actuators and waits until joints converge rather than a fixed time.

    pos_tol: 0.0087 rad (~0.5 deg) maximum joint position error.
    vel_tol: 0.02 rad/s maximum joint velocity across all arm joints.
    """
    arm_n = len(q_target)
    data.ctrl[:arm_n] = q_target
    data.ctrl[-1] = gripper
    steps = int(max_seconds / model.opt.timestep)
    every = max(1, int(1 / (fps * model.opt.timestep)))
    for k in range(steps):
        mujoco.mj_step(model, data)
        if record is not None and k % every == 0:
            record(data)
        pos_err = np.max(np.abs(data.qpos[:arm_n] - q_target))
        vel = np.max(np.abs(data.qvel[:arm_n]))
        if pos_err < pos_tol and vel < vel_tol:
            return True, (k + 1) * model.opt.timestep
    return False, max_seconds
```

- Joint interpolation moves the tool along a curve. For **vertical approach and lift**, interpolate in
  Cartesian space instead: solve IK at 5–10 intermediate points and chain them, each seeded from the previous one.
- Add settle time after descents and grasps. `SequenceAnimator` uses explicit "Wait" steps.
- Never jump `ctrl` across the workspace in one step.

## Pick-and-place recipe

```text
hover (+0.15 m, open) → descend (grasp height) → close → (settle) → lift (+0.15) → VERIFY grasp
→ transit (+0.15 over place) → lower (+0.05) → release → retreat (+0.15) → next / home
```

```python
def pick_and_place(model, data, pick, place, record=None, hover=0.15):
    up = np.array([0, 0, hover])
    plan = [("hover", pick + up, OPEN, 2.0), ("descend", pick, OPEN, 1.5), ("close", pick, CLOSED, 1.0),
            ("lift", pick + up, CLOSED, 1.5), ("transit", place + up, CLOSED, 2.0),
            ("lower", place + [0, 0, 0.05], CLOSED, 1.0), ("release", place + [0, 0, 0.05], OPEN, 0.8),
            ("retreat", place + up, OPEN, 1.0)]
    for name, pos, grip, dur in plan:
        q = solve_ik(model, data, "tcp", pos)
        if q is None:
            return False, f"IK failed at '{name}'"
        move_to(model, data, q, grip, dur, record)
        if name == "lift" and not touching(model, data, "left_finger", "cube"):
            return False, "grasp lost after lift"
    return True, "ok"
```

Grasp height = the detected top surface minus about half the object height (a 4 cm cube needs −2 cm).
Place height = container surface plus half the object height plus a small margin. Verify afterwards with
`touching(model, data, "cube", "tray")` or the object's pose. For ER-based verification, see the
`gemini-robotics-er` skill's video progress reference.

## Grippers

| Gripper | Actuator | Closed | Open |
|---|---|---|---|
| Menagerie Franka Panda | `actuator8` (tendon `split`), ctrlrange `[0, 255]` | `0` | `255` |
| Robotiq 2F-85 (Menagerie) | `fingers_actuator`, `[0, 255]` | `255` | `0` |

The second row is the opposite convention. **Always check the `ctrlrange` and direction** with a 1 s test closure in a fresh sim.

## The Franka demo's analytical Franka IK

`IkSystem.solve(pos, quat, currentQ)` calls `calculateAnalyticalIK(transform, q7)` from `FrankaAnalyticalIK.ts`:

1. Try the current q7, then scan q7 ± 0.5 rad in 0.1 steps. If that finds nothing, scan the full range in 0.2 steps.
2. Score every solution with `1.0·|q−q_cur|² + 0.05·|q−q_neutral|²` and keep the minimum.
3. It returns `null` when no solution exists. `SequenceAnimator` then holds position and logs `IK failed for step N`.

It is fast and deterministic, but **Panda-only** and it ignores collisions. For other robots, use DLS.

## Debugging IK

| Symptom | Cause / fix |
|---|---|
| Always `None` | Target out of reach, wrong site, or an impossible orientation. Check with position-only IK. |
| Arm moved but missed by cm, with no error | Silent fallback or checked post-motion: solver stopped at workspace boundary but caller ignored residual and drove actuators anyway. Enforce residual check *before* motion and raise `UnreachableError`. |
| Converges but the arm flips | Different IK branch. Seed from the current joints, add a null-space rest-pose bias. |
| Wrist spins wildly | Orientation near singular. Raise `damping` (0.05), lower `step`. |
| Tool tilted | Quaternion order mix-up (wxyz vs xyzw) or a wrong `DOWN` |
| Reaches in IK, not in sim | Joint `ctrlrange` narrower than `jnt_range`, gravity sag (stiffer gains, more settle time), or collision |
