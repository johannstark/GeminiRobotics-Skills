---
name: gemini-robotics-er
description: Builds robot perception, planning and verification with Gemini Robotics-ER 2 (gemini-robotics-er-2-preview, gemini-robotics-er-2-streaming-preview) through the Google Gen AI SDK 2.x Interactions API and Live API. Use when code calls Gemini Robotics / ER models, or for pointing, bounding boxes, trajectories, grasp points, 2D-to-3D lifting, webcam RGB grounding with ArUco homography and deparallax, SO-101 and LeRobot low-DOF arms, agentic vision (code execution), task orchestration with robot function calling, FastAPI/Gradio viewer integration, video moment finding / progress classification / success detection, real-time Live API robot agents, or migrating generateContent / ER 1.6 code.
---

# Gemini Robotics-ER 2

Gemini Robotics-ER (embodied reasoning) is a vision-language model for robots. It sees images,
video and audio. It returns text: JSON points and boxes, plans, or calls to your robot functions.
It never drives motors directly. **ER proposes, and deterministic code executes.**

Everything here was checked against the official docs (ai.google.dev, Oct 2026), the ER 2
notebook ([`gemini_robotics_er.ipynb`](https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb)), and `google-genai` 2.31 / `@google/genai` 2.30.

## 1. Pick the endpoint and API

| Model | API | Use for | Not supported |
|---|---|---|---|
| `gemini-robotics-er-2-preview` | **Interactions API** `client.interactions.create` (also legacy `generateContent`) | Pointing, boxes, trajectories, spatial logic, code execution, function-calling orchestration, video moment finding and progress | Live API |
| `gemini-robotics-er-2-streaming-preview` | **Live API** `client.aio.live.connect` / `ai.live.connect` (WebSocket) | Real-time agents: continuous video at ≤1 FPS, audio, BLOCKING tool calls | Structured output, code execution, caching, Interactions |
| `gemini-robotics-er-1.6-preview` | — | **Deprecated 31 Aug 2026.** Swap the model string. Video progress and streaming need ER 2. | |

Both ER 2 endpoints accept text, images, video and audio, and output text only. Limits: 131,072
input tokens and 65,536 output tokens. **Default thinking is `high`.** Levels are `minimal`,
`low`, `medium` and `high`. The docs recommend `medium` as the balance between latency and quality.

## 2. Choose the pattern

```
Need a one-shot target for a scripted motion?         -> Spatial query (points / boxes)    references/spatial-reasoning.md
RGB webcam + ArUco mat grounding, deparallax?         -> Real camera grounding             references/real-camera-grounding.md
Tiny text, gauges, counting, crop/zoom, annotate?     -> Agentic vision (code_execution)   references/agentic-vision.md
Multi-step task through your robot API, turn by turn? -> Orchestration (function calling)  references/orchestration.md
Did it work? When? How far along?                     -> Video understanding               references/video-progress.md
React continuously to a camera / voice, < 1 s loop?   -> Live API + streaming model        references/live-api.md
Test offline without API key (fakes, noisy model)?    -> Offline testing                   references/testing.md
Request/response shapes, SDK details, migration?      -> Interactions API reference        references/interactions-api.md
```

Tested scripts (`pip install "google-genai>=2.9.0" pydantic pillow opencv-python numpy`):
- `scripts/er_tools.py`: run `python gemini-robotics-er/scripts/er_tools.py --selftest` or import `parse_er_json`, `run_orchestration`, `LiveRobotSession`, `check_schema_hygiene`.
- `scripts/table_grounding.py`: run `python gemini-robotics-er/scripts/table_grounding.py --selftest` or import `compute_table_homography`, `solve_camera_pose`, `deparallax_point`.

## 3. Minimal correct calls

**Python** (`pip install "google-genai>=2.9.0" pydantic pillow opencv-python numpy`; reads `GEMINI_API_KEY`):

```python
from google import genai
client = genai.Client()

interaction = client.interactions.create(
    model="gemini-robotics-er-2-preview",
    input=[{"type": "user_input", "content": [                       # or a flat list of parts
        {"type": "image", "data": png_b64, "mime_type": "image/png"},  # media BEFORE the text
        {"type": "text", "text": "Point to the red cube. Points are [y, x] normalized to 0-1000."},
    ]}],
    generation_config={"thinking_level": "medium"},
    response_format={"type": "text", "mime_type": "application/json", "schema": {
        "type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
            "type": "object", "required": ["point", "label"], "properties": {
                "point": {"type": "array", "items": {"type": "integer"}}, "label": {"type": "string"}}}}}}},
)
items = json.loads(interaction.output_text)["items"]
```

**TypeScript** (`@google/genai >= 2.3.0`; request fields are **snake_case**; run it server-side):

```ts
import { GoogleGenAI } from '@google/genai';
const client = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const interaction = await client.interactions.create({
  model: 'gemini-robotics-er-2-preview',
  input: [{ type: 'image', data: pngB64, mime_type: 'image/png' }, { type: 'text', text: prompt }],
  generation_config: { thinking_level: 'medium' },
  response_format: { type: 'text', mime_type: 'application/json', schema },
});
const parsed = JSON.parse(interaction.output_text ?? '{}');
```

Read the response with `interaction.output_text` (final text), `interaction.steps` (typed
timeline: `thought`, `function_call`, `code_execution_call`, `code_execution_result`,
`model_output`), and `interaction.id` (pass it as `previous_interaction_id`).

## 4. Rules that prevent most bugs

1. **Coordinates are `[y, x]`, normalized 0–1000**, relative to the image you sent. Boxes are
   `[ymin, xmin, ymax, xmax]`. Map with the size of the image actually sent (`u = x/1000*W` and
   `v = y/1000*H`), never hardcoded dimensions. State the format in every prompt.
2. **Use the Interactions shapes.** `input=[{"type": "image" | "text" | "video", ...}]`,
   `generation_config={...}` and `response_format={...}`. Never pass `config=types.GenerateContentConfig(...)`,
   PIL images, or read `.text` / `.function_calls` on an Interaction. Those belong to `generateContent`.
3. **`tools`, `system_instruction` and `generation_config` are per-turn.** Resend them on every
   `interactions.create`, including turns that use `previous_interaction_id`.
4. **Don't use `thinking_budget`** with ER 2 / Gemini 3+. It can return 400 errors. Use `thinking_level`.
5. **Parse defensively, even with a schema.** Strip fences, slice to the outer JSON, drop
   duplicates and out-of-range coordinates, and cap the count (`er_tools.parse_er_json`).
6. **The capture camera must equal the back-projection camera**: same pose, intrinsics, crop
   and aspect ratio. Send only pristine, post-motion frames captured strictly after motion settles (never placeholders).
7. **Your robot functions return real outcomes.** On failure, send `is_error: true` with a
   reason so the model re-plans. Cap tool loops with `max_steps`. Validate every target against
   the workspace and joint limits before moving.
8. **Live API:** use only `"behavior": "BLOCKING"` tools, send at most 1 video frame per second,
   wait for `turn_complete` before the next heartbeat (otherwise barge-in cancels the action),
   and handle `tool_call_cancellation`.
9. **Keys:** unrestricted API keys get `403`, so add restrictions in AI Studio. Never ship a key
   to a browser. Use a backend proxy for Interactions, and **ephemeral tokens** for Live
   (they only work with Live).
10. **Storage:** interactions are stored by default (`store=true`: 55 days paid, 1 day free).
    `store=false` disables `previous_interaction_id`. Interactions is a Gemini Developer API
    feature, and `gs://` URIs aren't readable, so use the Files API.

## 5. Prompting that works (from the docs and notebook)

- Use plain language. If a term fails, try a synonym ("washer" or "ring").
- Say "DO NOT detect items that only match the description partially" to cut false positives.
- Cap counts ("no more than 25 items"). Ask for one point per query when you need a specific part.
- Generate prompt facts (axes, IDs, tool list) from code/config to prevent drift.
- Split complex tasks into separate prompts, and **run independent queries in parallel** (the
  notebook uses a ThreadPoolExecutor).
- For precision, **query several times and take the median** point.
- Resize images to about 800 px wide (the notebook does this) to save tokens. Crop or zoom for small objects.
- Thinking: use `low` for plain detection or pointing, and `high` for counting, payload or
  affordance reasoning, success detection and video. `medium` is the default compromise.
- With multiple images or video grids, **describe the layout in the prompt** ("first 4 images =
  start, last 4 = now"; "views left-to-right: top, wrist").

## 6. Lifting 2D to 3D (summary)

| Setup | Method |
|---|---|
| Browser sim (Three.js) | Raycast from the capture camera through NDC `(x/500-1, 1-y/500)`. See the `threejs-rendering` skill. |
| Python MuJoCo | Render depth from the same camera and deproject (`er_point_to_world` in `mujoco-physics/scripts/panda_pick_place.py`). Check segmentation for occlusion. |
| Real RGB-D | `Z = median depth`, `X = (u−cx)Z/fx`, `Y = (v−cy)Z/fy`, then `T_base←cam`. |
| Real RGB + ArUco mat | Homography to table plane z=0, IPPE camera pose, and deparallax for elevation h. See `references/real-camera-grounding.md`. |

A depth sample or raycast hits the top surface, so offset down to the grasp height. Box aspect gives
the gripper yaw: close across the short side. A 0–1000 integer grid quantizes to roughly
FOV/1000 per unit (about 1 mm/unit for a 1 m wide view).

## 7. In the Franka demo

The [Franka pick-and-place demo](https://ai.studio/apps/bundled/robotics_franka_pick_and_place) (React + Vite + `mujoco-js` + Three.js) is the reference app these skills were built from. File names below refer to it.

- `App.tsx::handleErSend` captures a ≤640 px PNG from a fixed top view and calls ER.
  It uses **SDK 1.x `ai.models.generateContent`** with `thinkingBudget: 0`. Points and
  box centres are raycast with `RenderSystem.project2DTo3D`. `SequenceAnimator` then runs a
  blind 15-step pick for every target. There is no verification, orchestration or Live API.
- The key is exposed to the browser through Vite `define` (`process.env.API_KEY`). That is fine
  for a demo but not for production.
- See `references/interactions-api.md#migrating-the-franka-demo` for the exact 1.x→2.x diff, and
  `references/orchestration.md` / `references/video-progress.md` for adding closed-loop behaviour.

## 8. Safety

ER can hallucinate or partially match objects. Treat every output as a proposal:

- Check workspace bounds, reachability (IK residual), collisions and velocity limits. An IK residual failure
  or out-of-bounds target **blocks the motion** and returns an actionable error; logging it is not enough.
- Require human confirmation for consequential actions, and keep a physical e-stop.
- Develop in simulation behind the same robot API before swapping in hardware.
- Camera and microphone data may capture people. Get consent, blur faces, and minimize retention.
  See <https://deepmind.google/models/gemini-robotics/safety>.
