# Gemini Robotics Agent Skills

Three [Agent Skills](https://agentskills.io) that teach coding agents (Claude Code, Antigravity, etc.) to build
robot manipulation software. A perception and planning model (**Gemini Robotics-ER 2**) drives a simulated robot
(**MuJoCo**, in the browser or in Python), which is visualized with **Three.js**.

| Skill | Use it for | Entry point |
|---|---|---|
| **gemini-robotics-er** | ER 2 via the Interactions API and Live API: pointing, boxes, trajectories, agentic vision, **task orchestration** with robot function calling, **video moment finding and progress classification**, real-time streaming agents | [`gemini-robotics-er/SKILL.md`](gemini-robotics-er/SKILL.md) |
| **mujoco-physics** | MuJoCo WASM and Python: loading Menagerie robots, state and control, cameras and deprojection, DLS IK, pick-and-place, episode video | [`mujoco-physics/SKILL.md`](mujoco-physics/SKILL.md) |
| **threejs-rendering** | Z-up scenes, geom→mesh, pose sync, perception snapshots, canvas video, ER point ↔ 3D, Python→browser streaming | [`threejs-rendering/SKILL.md`](threejs-rendering/SKILL.md) |

Each skill is a folder containing a short `SKILL.md` (loaded when relevant), `references/` (read on demand),
and, where useful, `scripts/` (run them, don't paste them).

```
├── install.sh
├── evals/
│   ├── gemini-robotics-er.json
│   ├── mujoco-physics.json
│   └── threejs-rendering.json
├── gemini-robotics-er/
│   ├── SKILL.md
│   ├── references/  interactions-api · spatial-reasoning · agentic-vision · orchestration · video-progress · live-api · real-camera-grounding · testing
│   └── scripts/
│       ├── er_tools.py          # helpers + offline selftest
│       └── table_grounding.py   # homography, IPPE camera pose, deparallax selftest
├── mujoco-physics/
│   ├── SKILL.md
│   ├── references/  python · wasm · ik-and-control · low-dof-arms
│   └── scripts/
│       ├── panda_pick_place.py  # 7-DOF Panda sim pipeline
│       └── so101_pick_place.py  # 5-DOF low-cost arm pipeline with residual checks
└── threejs-rendering/
    ├── SKILL.md
    ├── references/  geom-conversion · perception-and-video · remote-python · testing
    └── scripts/
        ├── export_meshes.py     # compiled mesh & manifest exporter with selftest
        └── test_viewer_smoke.py # headless Playwright viewer verification
```

The scenarios live in `evals/`; to run one, give the query and fixtures to a fresh agent once with the skills installed and once without, then check its output against `expected_behavior`. Run them before large edits to the skills.

## Install

```bash
git clone https://github.com/johannstark/GeminiRobotics-Skills.git ~/code/GeminiRobotics-Skills
cd ~/code/GeminiRobotics-Skills

./install.sh ~/code/my-robot-project          # symlinks into <project>/.claude/skills and <project>/.agents/skills
./install.sh --global                         # ~/.claude/skills and ~/.gemini/config/skills (every project)
./install.sh --copy ~/code/my-robot-project   # copy instead of symlink (Windows, containers, vendoring)
```

Symlinks point back to your clone, so `git pull` updates everything. Restart the agent session after installing.

| Agent | Project location | Global location | Invoke |
|---|---|---|---|
| Claude Code | `.claude/skills/<name>/SKILL.md` | `~/.claude/skills/<name>/` | Automatic by description, or `/gemini-robotics-er` |
| Antigravity (app / IDE) | `.agents/skills/<name>/SKILL.md` | `~/.gemini/config/skills/<name>/` | Automatic, or `/<skill-name>` |
| Antigravity CLI | `.agents/skills/` | `~/.gemini/antigravity-cli/skills/` | Automatic, or `/<skill-name>` |

>[!Note]
>Feel free to open a PR to add support for your coding agent

## How the pieces fit

```
camera image / video ──► Gemini Robotics-ER 2 ──► [y,x] points / boxes / tool calls
   ▲  (Three.js snapshot or                         │
   │   mujoco.Renderer)                             ▼
   │                                    2D → 3D (raycast / depth deprojection)
   │                                                │
   │                                    validate → IK → interpolated ctrl
   │                                                │
   └────────── MuJoCo (WASM / Python) ◄─────────────┘
                     │
                     └─► episode video ──► ER moment finding / progress ──► retry or done
```

| Goal | Skills to combine |
|---|---|
| Browser demo: click "detect", robot picks | ER spatial + Three.js perception + MuJoCo WASM |
| Python research loop with verification | MuJoCo Python script + ER orchestration + ER video progress |
| Real-time voice or camera agent | ER Live API + MuJoCo (Python or WASM) |
| Render a Python sim on the web | MuJoCo Python + Three.js remote-python |
| Cheap real arm + webcam + ArUco mat | ER real-camera grounding + MuJoCo low-DOF + ER testing |
| Python sim inside a Gradio/FastAPI app | MuJoCo Python (web server section) + Three.js remote-python |

### The Franka demo

Sections titled "In the Franka demo" refer to the
[Franka pick-and-place app](https://ai.studio/apps/bundled/robotics_franka_pick_and_place), a React + Vite +
`mujoco-js` + Three.js app built in AI Studio. These skills were developed from it. It captures a top-down snapshot,
asks ER 2 for points or boxes, raycasts them to 3D, and runs a scripted pick-and-place with analytical Franka IK.
The skills explain how to extend it with SDK 2.x, orchestration, video verification and the Live API.

## Quick checks

```bash
# offline logic tests for the ER helpers, TargetLock, and mock sessions (no API key)
python gemini-robotics-er/scripts/er_tools.py --selftest

# real-camera grounding: homography, IPPE camera pose, and deparallax
python gemini-robotics-er/scripts/table_grounding.py --selftest

# 7-DOF Panda sim pipeline: scene → render → ER-format point → 3D → IK → pick-and-place → verify
python mujoco-physics/scripts/panda_pick_place.py --menagerie mujoco_menagerie --video episode.mp4

# 5-DOF low-cost arm (SO-101) sim: ray-plane deprojection, 5-DOF IK, settle-on-convergence (exit 0)
MUJOCO_GL=egl python mujoco-physics/scripts/so101_pick_place.py --menagerie mujoco_menagerie --trials 10 --seed 0

# compiled mesh & manifest exporter (verifies compiled geometry, caching, geom branching)
python threejs-rendering/scripts/export_meshes.py --selftest

# headless Playwright viewer smoke test (verifies non-blank canvas, mesh count, quaternion order)
python threejs-rendering/scripts/test_viewer_smoke.py
```

## How these skills were verified (Oct 2026)

- **Docs:** [ai.google.dev](https://ai.google.dev/gemini-api/docs/robotics-overview) robotics overview, spatial, agentic, orchestration, streaming and video-progress pages;
  the Interactions API (overview, function calling, structured output, thinking, video); the Live API (capabilities,
  tools, sessions, ephemeral tokens); and Antigravity's skills docs.
- **Notebook:** the official
  [ER 2 notebook](https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb),
  sections 1–5. Prompts and outputs are quoted verbatim.
- **Code:** run against `google-genai` 2.32.0, `mujoco` 3.16.0, `@mujoco/mujoco` 3.16,
  `mujoco-js` 0.0.7, `three` 0.164.1/0.186, `playwright` 1.64.0, `websockets` 16.1.1,
  `fastapi` 0.162.0, `uvicorn` 0.54.0, `gradio` 6.30.0, and `httpx` 0.28.1.
  - Request bodies were validated against the SDK's own request models.
  - MuJoCo simulations executed headlessly (`MUJOCO_GL=egl`) with Menagerie `franka_emika_panda` and `robotstudio_so101`.
  - Three.js rendering verified in headless Chromium via Playwright (`test_viewer_smoke.py`).
  - Transport snippets in `threejs-rendering/references/remote-python.md` executed end-to-end: the WebSocket server streamed binary pose snapshots (`<II` header + `float32` `xpos`/`xquat`) from `robotstudio_so101` to a `websockets` client, and the FastAPI + Gradio app was verified with Starlette `TestClient` (`/viewer` `Cache-Control: no-store` and `/` Gradio mount without shadowing) plus a live `uvicorn` server streaming `/video_feed` multipart JPEGs to `httpx` with clean generator exit on disconnect.
  - The Live API MuJoCo loop in `gemini-robotics-er/references/live-api.md` was executed offline against `FakeLiveClient`/`FakeLiveSession` (`er_tools.py`) driving the headless `panda_pick_place.py` scene with EGL rendering (confirming tool execution, post-action JPEG frame before `send_tool_response`, and `turn_complete` handling with no pending tool call).
- **Evaluations (`evals/`):**
  Tested across 9 scenarios using fresh agent sessions with and without the skills (Claude Haiku and Opus, one run per cell, graded manually, 2026-10-21). Baseline agents scored 27/37 (Haiku) and 29/37 (Opus) without the skills, rising to 35/37 (Haiku) and 37/37 (Opus) with the skills installed. Scenarios live in `evals/`; run them by providing the query and fixtures to a fresh agent and checking outputs against `expected_behavior`.
- **Not executed (checked against docs and SDK types only):** live calls to paid ER cloud endpoints (`interactions.create`, `aio.live.connect` against `gemini-robotics-er-2-streaming-preview`, `files.upload`, `authTokens.create`), Live API `part.thought` streaming behavior and server-side barge-in on video vs text/audio, and the browser polling/smoothing/marker TypeScript snippets in `remote-python.md`.

### Common mistakes these skills prevent

- **IK-frame origin:** Measuring table corners or targets from the pan rotation axis rather than the kinematic base root (causes 35–45 mm X-offset grasping errors on offset-base arms).
- **Camera HFOV & parallax:** Using nominal sim FOV for real lenses, or ignoring parallax on elevated objects (a 100 mm height produces > 15 mm spatial error if uncorrected).
- **Grasp physics & qpos teleporting:** Teleporting arm or gripper joints via `qpos` (leaves `qvel=0`, eliminating contact friction and causing lifted objects to slip). Always command actuators via `ctrl`.
- **Gripper force saturation:** Over-torquing gripper actuators (causes objects to squeeze out) or lifting before gripper servos stall.
- **Silent IK fallback:** Permitting arm motion on unconverged IK results or substituting position-only solutions without checking residuals. Always raise `UnreachableError` before motion.
- **Sim steppers in web routes:** Calling `mj_step` inside web endpoints or request handlers alongside a physics loop (causes sim time to outpace real time).
- **Compiled meshes vs raw STLs:** Loading raw CAD files in Three.js (MuJoCo automatically re-centers and re-orients meshes at compile time; loading raw STLs causes link misalignments). Always export compiled `mesh_vert`/`mesh_face`.
- **Viewer state polling:** Polling state endpoints without an in-flight guard, monotonic sequence numbers, or reset epoch detection (causes jerky out-of-order motion and sliding on resets).
- **Prompt/controller contradictions:** Prompts promising footprint grounding while controllers pass raw object points, or prompts requiring single physical actions while controllers batch multiple.
- Passing `config=GenerateContentConfig(...)` to `client.interactions.create`. Use `generation_config` /
  `response_format` / `tools` / `system_instruction`, and resend them on every turn.
- Reading `.text` / `.function_calls` on an Interaction. Read `output_text` and `steps`.
- `thinking_budget` on ER 2 (it can return 400). Use `thinking_level`.
- Live API: setting `behavior` on the `Tool` instead of on each function declaration; using
  `session.send` / `media_chunks`; heartbeats on a blind timer, whose barge-in cancels tool calls.
- JS ephemeral tokens without `apiVersion: 'v1alpha'`.
- MuJoCo WASM 3.16: `new DoubleBuffer([..])` and `.getView()`. Use `new DoubleBuffer(n)` and `.GetView()`.
  `mj_ray` takes 9 arguments.
- MJCF angles defaulting to degrees; Menagerie keyframes after adding free bodies; mocap welds without an explicit
  `relpose`; overhead cameras occluded by the arm.
- Three.js render targets are linear, so perception images need an sRGB conversion.

## Keeping them current

- The model IDs are previews. Check <https://ai.google.dev/gemini-api/docs/robotics-overview> for new endpoints and
  deprecations.
- After upgrading an SDK, rerun the quick checks. For JS WASM changes, compare against
  `node_modules/@mujoco/mujoco/mujoco.d.ts`.
- Keep each `SKILL.md` short. Put new detail in `references/`, and link to it from the decision tree.

***

Made in Colombia 🇨🇴 with Love ❤️
