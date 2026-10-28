# Task orchestration: ER calls your robot API (notebook section 4)

ER 2 breaks a goal into steps and calls **your** functions (move, grasp, VLA skills, speak, ...).
Your code executes each call, returns the real outcome, and the model plans the next step. The source is
<https://ai.google.dev/gemini-api/docs/robotics-orchestration> and notebook section 4. For a real-time
version, see `live-api.md`.

## Contents
- [The official example (verbatim)](#the-official-example-verbatim)
- [What the model actually did](#what-the-model-actually-did)
- [Production loop](#production-loop)
- [Wiring tools to a simulator or robot](#wiring-tools-to-a-simulator-or-robot)
- [Closed loop with fresh images](#closed-loop-with-fresh-images)
- [Designing the robot API](#designing-the-robot-api)
- [ER proposes, the controller measures](#er-proposes-the-controller-measures)
- [The prompt is a contract the controller enforces](#the-prompt-is-a-contract-the-controller-enforces)
- [Fine-grained tools vs smart stateless primitives](#fine-grained-tools-vs-smart-stateless-primitives)
- [The verification ladder](#the-verification-ladder)
- [TypeScript port](#typescript-port)
- [legacy generateContent version](#legacy-generatecontent-version)

## The official example (verbatim)

SO-101 arm, a blue block and an orange bowl. Coordinates are relative to an origin given in normalized image space.

```python
def move(x, y, high):
    print(f"Mock Robot: Moving to coordinates: {x}, {y}, {'high above table' if high else 'down at table level'}")

def setGripperState(opened):
    print(f"Mock Robot: {'Opening gripper' if opened else 'Closing gripper'}")

robot_origin_y = 300
robot_origin_x = 500

move_function = {
    "type": "function",
    "name": "move",
    "description": "Moves the arm to the given coordinates.",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate relative to the origin"},
            "y": {"type": "integer", "description": "Y coordinate relative to the origin"},
            "high": {"type": "boolean", "description": "Set to True to lift the robot arm above the scene for avoiding obstacles. Set to False to place the gripper on the surface."}
        },
        "required": ["x", "y", "high"]
    }
}

set_gripper_state_function = {
    "type": "function",
    "name": "setGripperState",
    "description": "Opens or closes the robot's gripper.",
    "parameters": {
        "type": "object",
        "properties": {
            "opened": {"type": "boolean", "description": "True opens the gripper, False closes the gripper."}
        },
        "required": ["opened"]
    }
}

prompt = (
    "You are a robotic arm with six degrees-of-freedom. "
    f"The origin point for calculating the moves is at normalized point y={robot_origin_y}, x={robot_origin_x}. "
    "Use this as the new (0,0) for calculating moves, allowing x and y to be negative.\n\n"
    "Find the blue block and the orange bowl. Calculate their coordinates relative to the origin.\n"
    "Perform a pick and place operation where you pick up the blue block and place it into the orange bowl. "
    "Call the appropriate sequence of functions to complete this operation."
)

interaction = client.interactions.create(
    model=MODEL_ID,
    input=[{"type": "user_input", "content": [
        {"type": "image", "data": img_b64, "mime_type": "image/png"},
        {"type": "text", "text": prompt}
    ]}],
    tools=[move_function, set_gripper_state_function],
    generation_config={"thinking_level": "low"}
)

max_steps = 15  # Safety limit to prevent infinite tool-calling loops
step_count = 0
while step_count < max_steps:
    step_count += 1
    tool_calls = [step for step in interaction.steps if step.type == "function_call"]
    if not tool_calls:
        print("Sequence complete.")
        if interaction.output_text:
            print(f"Model Summary: {interaction.output_text}")
        break

    function_results = []
    for step in tool_calls:
        if step.name == "move":
            move(**step.arguments)
        elif step.name == "setGripperState":
            setGripperState(**step.arguments)
        function_results.append({
            "type": "function_result",
            "name": step.name,
            "call_id": step.id,
            "result": [{"type": "text", "text": '{"status": "success"}'}]
        })

    # previous_interaction_id keeps history server-side; tools must be resent every turn
    interaction = client.interactions.create(
        model=MODEL_ID,
        previous_interaction_id=interaction.id,
        tools=[move_function, set_gripper_state_function],
        input=function_results
    )
```

## What the model actually did

Notebook run:

```text
Opening gripper → move(160, 435, high) → move(160, 435, low) → Closing gripper → move(160, 435, high)
→ move(-250, 70, high) → Opening gripper → move(0, 200, high) → "Sequence complete."
```

Docs run:

```text
Opening → move(160,440,high) → move(160,440,low) → Closing → move(160,440,high) → move(-250,60,high)
→ move(-250,60,low) → Opening → move(-250,60,high)
```

Two lessons:

1. **Plans vary between runs.** In the notebook run the model released the block while "high above
   table", so it dropped the block. Your executor must enforce invariants such as "lower before release"
   and "never close on air", or the prompt must state them. Treat the plan as a proposal.
2. **Argument semantics are yours to define.** Here `x` and `y` are offsets from a normalized origin, in
   image axes (x right, y down). Absolute point = `[origin_y + y, origin_x + x]`. Say which way each
   axis points in the prompt.

## Production loop

`scripts/er_tools.py::run_orchestration` implements the loop with these extras:

- Real results go back to the model. Failures carry `is_error: True`, so the model can recover.
- Exceptions raised by your executor are reported to the model instead of crashing the loop.
- `tools`, `generation_config` and `system_instruction` are resent every turn.
- A step cap is enforced, and the function returns `{"done", "summary", "calls", "interaction_id"}`.

```python
from er_tools import run_orchestration, user_input, image_part, text_part, encode_image

b64, mime = encode_image(rgb)              # rgb from the sim or camera
result = run_orchestration(
    client, user_input(image_part(b64, mime), text_part(prompt)),
    tools=[move_function, set_gripper_state_function],
    execute=robot.execute,                 # (name, args) -> dict, raises or returns {"status": "error", ...}
    max_steps=15,
    system_instruction="Always lower the gripper to the surface before opening it to place an object.",
)
```

## Wiring tools to a simulator or robot

This code was tested: it replayed the docs' plan through `run_orchestration` against the MuJoCo Panda, and the
cube ended in the tray. An out-of-image `move` was returned to the model as `is_error: true`.

```python
from panda_pick_place import (build_scene, render_rgbd, er_point_to_world, solve_ik, move_to,
                              touching, OPEN, CLOSED)        # mujoco-physics/scripts

def in_workspace(p, lo=(0.2, -0.6, 0.0), hi=(0.8, 0.6, 0.6)):
    return all(l <= v <= h for v, l, h in zip(p, lo, hi))

class SimRobot:
    """Executes ER's move/setGripperState on the MuJoCo Panda from mujoco-physics/scripts/panda_pick_place.py."""
    def __init__(self, model, data, renderer, origin_yx=(300, 500), cam="top"):
        self.m, self.d, self.r, self.origin, self.cam = model, data, renderer, origin_yx, cam
        _, self.depth, _ = render_rgbd(renderer, data, cam)        # same camera + moment as the image sent to ER
        self.grip = OPEN

    def execute(self, name, args):
        if name == "setGripperState":
            self.grip = OPEN if args["opened"] else CLOSED
            move_to(self.m, self.d, self.d.ctrl[:7].copy(), self.grip, 0.8)   # hold pose, actuate gripper
            return {"status": "success", "gripper": "open" if args["opened"] else "closed"}
        if name == "move":
            yx = [self.origin[0] + args["y"], self.origin[1] + args["x"]]       # relative -> absolute
            if not all(0 <= c <= 1000 for c in yx):
                return {"status": "error", "reason": f"point {yx} outside the camera image"}
            p = er_point_to_world(self.m, self.d, self.cam, yx, self.depth)
            p[2] = p[2] + 0.15 if args["high"] else max(p[2] - 0.02, 0.02)     # transit height vs grasp height
            if not in_workspace(p):
                return {"status": "error", "reason": f"target {p.round(3).tolist()} outside workspace"}
            q = solve_ik(self.m, self.d, "tcp", p)
            if q is None:
                return {"status": "error", "reason": "unreachable (IK failed)"}
            move_to(self.m, self.d, q, self.grip, 1.5)
            return {"status": "success", "tcp": self.d.site("tcp").xpos.round(3).tolist(),
                    "holding_object": bool(self.grip == CLOSED and touching(self.m, self.d, "left_finger", "cube"))}
        return {"status": "error", "reason": f"unknown function {name}"}
```

Checklist:

1. Convert coordinates exactly as the prompt defines them (relative or absolute, `[y, x]`, 0–1000).
2. Lift to 3D using the **same camera** that produced the image (see the spatial reasoning reference).
3. Map `high` to a safe transit height, and `low` to grasp or place height.
4. Validate: workspace box, IK residual, joint limits, collisions on a scratch `MjData`, speed.
5. Execute, wait for motion to settle, then **return measured state**, not just "success".
   Include the gripper state, whether it's holding an object, and the final pose.
6. Enforce invariants in code: lower before release, and don't close on air.

## Closed loop with fresh images

`function_result.result` can include images (Gemini 3+ multimodal function responses), so the
model can see the effect of each action:

```python
frame_b64, _ = encode_image(capture_rgb(), width=800)
results.append({"type": "function_result", "name": step.name, "call_id": step.id, "result": [
    {"type": "text", "text": json.dumps(outcome)},
    {"type": "image", "mime_type": "image/png", "data": frame_b64},   # scene after the action
]})
```

The alternative is a new user turn per step, `[image, "Goal: ... What next?"]`, with `previous_interaction_id`.
Either way, add a final verification step: multi-view before/after success (spatial-reasoning.md) or
video progress (video-progress.md).

## Designing the robot API

- Prefer **few, high-level, idempotent** functions (`pick(y, x)`, `place(y, x)`, `go_home()`,
  `run_skill(name)` for a VLA) over joint-level ones. ER plans; your controllers move.
- Use precise descriptions, units and coordinate frames. Use enums for discrete choices.
- Add an explicit `task_complete(summary)` or `report_failure(reason)` function if you need a structured end state.
- For multi-robot setups, give each robot its own namespace (`left_arm_move`, `right_arm_move`) or a `robot` enum.
- Keep the step cap (`max_steps`) and a wall-clock timeout. Log every call and result.

## ER proposes, the controller measures

ER's visual pointing drifts by roughly $25\text{ mm}$ between consecutive calls at a $5\text{ cm}$ hover height.
Attempting to close a fine-grained visual servoing loop with ER prompts fails:
- **Failed prompt approaches**:
  1. Asking ER for prompted `[dy, dx]` millimeter offsets.
  2. A report-only ER visual fingertip locator.
  3. A verify-and-correct visual feedback loop with `visual_offset_delta`.
- **What actually works**: ER selects *which object* to interact with and proposes an approximate $[y, x]$ coordinate.
  Physical geometry and sensors measure millimeters:
  1. **Snap ER points**: Snap ER's proposal to a 2D CV-detected contour (color blob or segmentation) within an
     $80/1000$ normalized radius ($80/1000$ bounds typical ER pointing jitter of $\sim 25\text{ mm}$ at $5\text{ cm}$ hover).
  2. **Lock identity in metric space**: Project the pre-motion detection to table-plane millimeters ($Z = 0$),
     lock the object identity, and match subsequent proposals within a $40\text{ mm}$ metric radius (object radius
     plus motion margin).
  3. **Echo canonical point**: Echo the snapped canonical coordinate back to the model in the tool response.
  4. **Measured fields override visual estimates**: When the gripper carries an ArUco fiducial tag, FK combined
     with ray-plane deparallax measures millimeters directly. In the system prompt, state:
     `"Measured fields in tool results override your visual estimate."`
  5. **Explicit next action**: Returning an explicit `"next"` hint in tool failure results
     (e.g., `"next": "move_to_point(point=[412, 518], z_mm=30)"`) enables the model to recover
     from workspace and IK refusals reliably.
  6. **Single source of truth**: Generate tool descriptions in the prompt and function declarations from a single
     code registry. Generate physical facts (table axes, marker IDs) from hardware configuration to eliminate drift.
  7. **Schema hygiene**: Run `check_schema_hygiene(tools, handlers)`. If a handler reads an argument key not declared
     in its schema, the model will never emit it, and downstream code will silently receive `None` or a default.

## The prompt is a contract the controller enforces

Never rely on prompt instructions that the controller does not strictly enforce in code:
- **Single pointing contract**: Specify exactly one pointing rule. Do not contradict instructions by asking for
  "the object center" in one sentence and "the exact bottom contact point" in another.
- **One physical action per turn**: Prompting the model to perform "one physical action per turn" is not enough;
  the controller must enforce it. If the model returns multiple physical tool calls in a single turn, execute the
  first physical call and return:
  `{"status": "error", "reason": "one physical action per turn; re-issue after observing"}` with `is_error: True`
  for the remaining calls.
- **Plan → Validate → Execute loop**: Validate every parameter (schema types, 3D workspace bounding box,
  IK feasibility and residual tolerance, and camera frame freshness) *before* issuing actuator commands.
  Refuse motion immediately with an actionable error if any validation fails.

## Fine-grained tools vs smart stateless primitives

| Paradigm | Tools | Pros / Cons | Recommendation |
|---|---|---|---|
| Fine-grained primitives | `move(point, z)`, `open()`, `close()` | High planning flexibility, but forces complex cross-turn target lock state that churns (`no_lock` → `retry_exhausted`). | Use only on high-accuracy robots with reliable real-time Cartesian controllers. |
| Smart stateless primitives | `pick(point)`, `place(point)`, `inspect(point)` | Robust and self-contained: detects, aligns, descends vertically, verifies grip, and retracts within one call. | **Default for low-cost arms (e.g. SO-101)**. Keeps ER at the high-level task planning layer. |

### Low-DOF toolset example (SO-101 style)

```python
SO101_TOOLS = [
    {
        "type": "function",
        "name": "pick_object",
        "description": "Picks an object at normalized [y, x] (0-1000). Aligns top-down, descends, grips, and lifts.",
        "parameters": {
            "type": "object",
            "properties": {
                "y": {"type": "integer", "description": "Normalized Y (0-1000)"},
                "x": {"type": "integer", "description": "Normalized X (0-1000)"},
            },
            "required": ["y", "x"],
        },
    },
    {
        "type": "function",
        "name": "place_object",
        "description": "Places the held object at normalized [y, x] (0-1000). Moves high, descends, opens, and lifts.",
        "parameters": {
            "type": "object",
            "properties": {
                "y": {"type": "integer", "description": "Normalized Y (0-1000)"},
                "x": {"type": "integer", "description": "Normalized X (0-1000)"},
            },
            "required": ["y", "x"],
        },
    },
]
```

## The verification ladder

Verify manipulation outcomes through a progression of increasingly rich, geometry-aware checks:

```
  Level 4: ER video progress classification (video-progress.md)
     ▲
  Level 3: ER multi-view before/after verification (spatial-reasoning.md)
     ▲
  Level 2: Geometry-aware vision veto (lift >= 5 cm before checking 2D overlap)
     ▲
  Level 1: Gripper aperture sensor (empty-jaw angle + 6 deg threshold)
```

1. **Gripper aperture sensor (Level 1)**:
   Compare the gripper encoder angle against the closed-on-empty calibration. If $\theta \le \theta_{\text{empty}} + 6^\circ$,
   the jaws have closed on air. Fail immediately without burning vision tokens.
2. **Geometry-aware vision veto (Level 2)**:
   Vision may only veto when the 3D geometry permits reliable distinction. A cube lifted only $3\text{ cm}$
   still heavily overlaps its table footprint in an oblique camera view (IoU up to $0.6$). At $5\text{ cm}$ lift,
   the IoU drops to $\le 0.3$. **Lift the arm at least $5\text{ cm}$ before evaluating visual grasp success.**
3. **Pristine, post-motion frames**:
   - Send only pristine, unannotated camera images to the model (keep debug bounding boxes and markers in a separate
     operator buffer).
   - Capture observation frames strictly **after** robot motion settles, verified by timestamp.
   - Fail closed if no fresh frame is received within $0.5\text{ s}$ (`FRAME_FRESHNESS_TIMEOUT_S`). Never send placeholder frames.

## TypeScript port

```ts
let it = await client.interactions.create({
  model: 'gemini-robotics-er-2-preview', tools: TOOLS, generation_config: { thinking_level: 'low' },
  input: [{ type: 'image', data: imgB64, mime_type: 'image/png' }, { type: 'text', text: prompt }],
});
for (let i = 0; i < 15; i++) {
  const calls = (it.steps ?? []).filter((s: any) => s.type === 'function_call');
  if (!calls.length) { console.log('Summary:', it.output_text); break; }
  const results = [];
  for (const c of calls as any[]) {
    let out: any, isError = false;
    try { out = await robot.execute(c.name, c.arguments); isError = out?.status === 'error'; }
    catch (e) { out = { status: 'error', reason: String(e) }; isError = true; }
    results.push({ type: 'function_result', name: c.name, call_id: c.id, is_error: isError,
                   result: [{ type: 'text', text: JSON.stringify(out) }] });
  }
  it = await client.interactions.create({ model: 'gemini-robotics-er-2-preview', tools: TOOLS,
    generation_config: { thinking_level: 'low' }, previous_interaction_id: it.id, input: results });
}
```

For the Franka demo, `robot.execute('move', ...)` maps to `RenderSystem.project2DTo3D` plus `IkSystem.solve` plus
interpolated `ctrl` (the same building blocks `SequenceAnimator` uses). Keep the API key server-side.

## Legacy generateContent version

The docs also show the stateless `generateContent` loop. Use it only when you can't use Interactions:

```python
contents = [types.Content(role="user", parts=[types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                                             types.Part(text=prompt)])]
for _ in range(15):
    resp = client.models.generate_content(model=MODEL_ID, contents=contents, config=types.GenerateContentConfig(
        tools=[robot_tools], thinking_config=types.ThinkingConfig(thinking_level="low")))
    contents.append(resp.candidates[0].content)                      # keep thought signatures
    calls = [p for p in resp.candidates[0].content.parts if p.function_call]
    if not calls: break
    contents.append(types.Content(role="user", parts=[
        types.Part.from_function_response(name=p.function_call.name, response=execute(p.function_call.name, p.function_call.args))
        for p in calls]))
```
