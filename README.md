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
├── gemini-robotics-er/
│   ├── SKILL.md
│   ├── references/  interactions-api · spatial-reasoning · agentic-vision · orchestration · video-progress · live-api
│   └── scripts/er_tools.py          # helpers + offline selftest
├── mujoco-physics/
│   ├── SKILL.md
│   ├── references/  python · wasm · ik-and-control
│   └── scripts/panda_pick_place.py  # tested end-to-end sim pipeline
└── threejs-rendering/
    ├── SKILL.md
    └── references/  geom-conversion · perception-and-video · remote-python
```

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

### The Franka demo

Sections titled "In the Franka demo" refer to the
[Franka pick-and-place app](https://ai.studio/apps/bundled/robotics_franka_pick_and_place), a React + Vite +
`mujoco-js` + Three.js app built in AI Studio. These skills were developed from it. It captures a top-down snapshot,
asks ER 2 for points or boxes, raycasts them to 3D, and runs a scripted pick-and-place with analytical Franka IK.
The skills explain how to extend it with SDK 2.x, orchestration, video verification and the Live API.

## Quick checks

```bash
# offline logic tests for the ER helpers (no API key)
python gemini-robotics-er/scripts/er_tools.py --selftest

# full sim pipeline: scene → render → ER-format point → 3D → IK → pick-and-place → verify → MP4
pip install mujoco numpy pillow "imageio[ffmpeg]"
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
python mujoco-physics/scripts/panda_pick_place.py --menagerie mujoco_menagerie --video episode.mp4   # exit 0 = success
```

## How these skills were verified (Oct 2026)

- **Docs:** [ai.google.dev](https://ai.google.dev/gemini-api/docs/robotics-overview) robotics overview, spatial, agentic, orchestration, streaming and video-progress pages;
  the Interactions API (overview, function calling, structured output, thinking, video); the Live API (capabilities,
  tools, sessions, ephemeral tokens); and Antigravity's skills docs.
- **Notebook:** the official
  [ER 2 notebook](https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb),
  sections 1–5. Prompts and outputs are quoted verbatim.
- **Code:** run against `google-genai` 2.31, `@google/genai` 2.30, `mujoco` 3.16, `@mujoco/mujoco` 3.16,
  `mujoco-js` 0.0.7, `three` 0.186 and `websockets` 15.
  - Request bodies were validated against the SDK's own request models.
  - MuJoCo and WASM snippets were executed with the Menagerie Panda.
- **Not executed:** live calls to the ER endpoints and browser-only APIs (WebGL readback, `MediaRecorder`). These
  follow the official docs and the SDK sources.

### Common mistakes these skills prevent

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
