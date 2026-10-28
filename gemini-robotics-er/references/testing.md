# Testing robot pipelines without the API

## Contents
- [1. Testing pyramid for robotics agents](#1-testing-pyramid-for-robotics-agents)
- [2. Fake Live API sessions with real SDK types](#2-fake-live-api-sessions-with-real-sdk-types)
- [3. Fake Interactions client for orchestration](#3-fake-interactions-client-for-orchestration)
- [4. Scripted noisy pointing against simulator ground truth](#4-scripted-noisy-pointing-against-simulator-ground-truth)
- [5. Hardware perturbation: hidden-offset robots](#5-hardware-perturbation-hidden-offset-robots)
- [6. Units and silent defaults](#6-units-and-silent-defaults)
- [7. Test coverage honesty](#7-test-coverage-honesty)

---

## 1. Testing pyramid for robotics agents

Live ER API calls are non-deterministic, cost tokens, and require network access and physical hardware.
A mock-only test suite that tests only superficial route existence easily stays green while the real
control loop is completely broken.

Test the robotics stack offline through four complementary layers:
1. **Schema & contract checks**: Verify declared tool schemas against handler parameter signatures (`check_schema_hygiene`)
   and prompt contracts (see the example lint `lint_prompt_rules`).
2. **Fake server streams**: Replay realistic sequences of `google.genai.types.LiveServerMessage` objects through the
   client receive loop.
3. **Noisy synthetic models**: Drive the real robot controllers in MuJoCo physics using a synthetic vision model
   that samples points with realistic Gaussian jitter ($\sigma = 8/1000$).
4. **Hardware perturbation**: Inject known physical distortions (uncalibrated camera, encoder pan sag) to ensure
   the pipeline fails closed or detects the fault.

## 2. Fake Live API sessions with real SDK types

The Live API receive loop processes typed protobuf-backed server messages. Rather than generic mocks,
test the receive loop with `FakeLiveSession` (available in `scripts/er_tools.py`) populated with real
`google.genai.types` objects:

```python
from google.genai import types
from er_tools import FakeLiveSession

messages = [
    types.LiveServerMessage(
        session_resumption_update=types.LiveServerSessionResumptionUpdate(
            new_handle="resume_handle_001", resumable=True
        )
    ),
    types.LiveServerMessage(
        tool_call=types.LiveServerToolCall(
            function_calls=[
                types.FunctionCall(
                    id="call_1",
                    name="pick_object",
                    args={"y": 450, "x": 520},
                )
            ]
        ),
        server_content=types.LiveServerContent(turn_complete=True),
    ),
]

session = FakeLiveSession(messages)
# Test your robot receive loop directly:
async for msg in session.receive():
    if msg.tool_call:
        ...
```

This verifies that `turn_complete` combined with a `tool_call` does not prematurely stop the loop,
and that resumption tokens are correctly persisted.

## 3. Fake Interactions client for orchestration

For turn-by-turn Interactions loops (`run_orchestration`), simulate multi-step planning with a scripted
fake that inspects previous turns and simulates recovery from errors:

```python
from types import SimpleNamespace as NS
from er_tools import run_orchestration, text_part, user_input

script = [
    # Turn 1: model attempts an invalid out-of-bounds reach
    [NS(type="function_call", name="move", id="c1", arguments={"x": 9999, "y": 0, "high": False})],
    # Turn 2: model observes error and issues a valid move
    [NS(type="function_call", name="move", id="c2", arguments={"x": 200, "y": 300, "high": True})],
    # Turn 3: model concludes sequence
    [NS(type="model_output")],
]

class FakeInteractions:
    def __init__(self):
        self.sent = []
    def create(self, **kw):
        self.sent.append(kw)
        idx = len(self.sent) - 1
        return NS(id=f"turn_{idx}", steps=script[idx], output_text="Done" if idx == 2 else None)

client = NS(interactions=FakeInteractions())
res = run_orchestration(client, user_input(text_part("run")), tools=[...], execute=my_executor)
assert res["done"] and len(res["calls"]) == 2
```

## 4. Scripted noisy pointing against simulator ground truth

ER visual pointing is not an exact coordinate measurement. In practice, visual pointing proposals exhibit
roughly $\sigma = 8/1000$ Gaussian noise around the true object center (about $25\text{ mm}$ drift at $5\text{ cm}$ hover).

Use `ScriptedNoisyModel` (in `scripts/er_tools.py`) to drive the **real** robot controller against
a headless MuJoCo simulation, and grade the final outcome against ground-truth simulator physics:

```python
from er_tools import ScriptedNoisyModel

# True cube position in 0-1000 normalized image coordinates
true_center_yx = [450, 500]
model = ScriptedNoisyModel(noise_sigma=0.008, seed=42)

for trial in range(10):
    proposed_yx = model.propose_point(true_center_yx)
    # The real pipeline should snap the noisy proposal to the CV blob or verify with FK
    result = robot_pipeline.pick_from_er_point(proposed_yx)
    assert result["status"] == "success"
```

## 5. Hardware perturbation: hidden-offset robots

Real low-cost servos (such as STS3215 on the SO-101) suffer from motor zero offsets, backlash,
and gravity sag. Forward kinematics (FK) alone does not see these errors.

Test perception and verification routines using `HiddenOffsetRobot` (in `scripts/er_tools.py`):
- Injects a hidden hardware offset (e.g. $1.5^\circ$ pan offset, producing a $5\text{–}8\text{ mm}$ physical miss).
- Forward kinematics reports nominal joint angles without the offset.
- The external camera feed observes the true physical arm position with the offset.
- **Pass criterion**: The verification ladder must detect that the gripper missed the target or failed to close,
  vetoing the action before motion proceeds.

## 6. Units and silent defaults

Naming and default values cause subtle bugs across sim and real hardware boundaries:
1. **Explicit unit suffixes**: Include units directly in variable and constant names:
   - Use `HOME_RAD` (radians) or `HOME_DEG` (degrees), never ambiguous `HOME`.
   - Use `z_mm` or `z_m`, never raw `z`.
2. **Assert plausible numerical ranges**:
   ```python
assert -np.pi <= q <= np.pi, f"Joint angle {q} appears to be in degrees, expected radians"
assert 0.0 <= z_m <= 1.0, f"Height {z_m} appears to be in millimeters, expected meters"
   ```
3. **No silent fallbacks to sim constants**:
   If a camera calibration or homography is missing in real mode, **raise immediately**. Never silently
   substitute hard-coded simulation camera positions or nominal matrix constants.

## 7. Test coverage honesty

Test documentation must strictly match what the tests actually verify:
- If a test only asserts that an HTTP route or HTML file exists, state that it is a route smoke test.
- Do **not** claim a test suite validates "viewer 3D asset integrity" or "mesh transformations" unless it
  actually parses mesh buffers or verifies canvas rendering (see `threejs-rendering` skill, `testing.md`).
