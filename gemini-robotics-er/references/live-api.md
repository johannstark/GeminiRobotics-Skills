# Real-time robot agents with the Live API (gemini-robotics-er-2-streaming-preview)

Sources: <https://ai.google.dev/gemini-api/docs/robotics-streaming>, the Live API capability,
session and tools guides, and the ephemeral tokens guide. Config shapes were validated with `google-genai` 2.31.

## Contents
- [Specs and constraints](#specs-and-constraints)
- [The three-part agent](#the-three-part-agent)
- [Complete Python agent driving MuJoCo](#complete-python-agent-driving-mujoco)
- [Heartbeat rules](#heartbeat-rules)
- [Long sessions](#long-sessions)
- [Speech through TTS as a tool](#speech-through-tts-as-a-tool)
- [Browser agent with ephemeral tokens](#browser-agent-with-ephemeral-tokens)
- [Pitfalls](#pitfalls)

## Specs and constraints

| Item | Value |
|---|---|
| Model | `gemini-robotics-er-2-streaming-preview` (Live only; the standard ER 2 doesn't support Live) |
| Protocol | Stateful WebSocket (WSS) |
| Input | Text; JPEG frames at **≤ 1 FPS** (768×768 recommended for Live); raw PCM audio, 16-bit, 16 kHz, little-endian |
| Output | **Text only**. Speak through an external TTS. |
| Tools | Function calling with **BLOCKING functions only** (for robotics). Google Search is available. No code execution, no structured output. |
| Thinking | Supported: `thinking_config=types.ThinkingConfig(thinking_level=...)` |
| Connection | About 10 min per WebSocket. Audio+video context lasts about 2 min without compression (audio-only about 15 min). |

## The three-part agent

1. **Declare robot capabilities as BLOCKING tools.** The model waits for each physical action to finish.
2. **Stream sensor input into one persistent session**: frames, audio, text.
3. **Handle `tool_call` in a receive loop**: execute against the robot, then `send_tool_response`.

Video frames alone don't trigger a turn. A turn needs text or audio. That is why proactive behaviour
uses a **heartbeat**.

## Complete Python agent driving MuJoCo

```python
import asyncio, io, os
import mujoco, numpy as np
from PIL import Image
from google import genai
from google.genai import types
from panda_pick_place import build_scene, render_rgbd, er_point_to_world, pick_and_place  # mujoco-physics/scripts

MODEL = "gemini-robotics-er-2-streaming-preview"
TOOLS = [{"function_declarations": [
    {"name": "pick_and_place", "behavior": "BLOCKING",
     "description": "Pick the object at image point (y, x) and place it at image point (place_y, place_x). "
                    "Coordinates are [y, x] normalized to 0-1000 in the most recent camera frame.",
     "parameters": {"type": "OBJECT", "properties": {
         "y": {"type": "INTEGER"}, "x": {"type": "INTEGER"},
         "place_y": {"type": "INTEGER"}, "place_x": {"type": "INTEGER"}},
         "required": ["y", "x", "place_y", "place_x"]}},
    {"name": "ack", "behavior": "BLOCKING", "description": "Nothing needs to be done right now.",
     "parameters": {"type": "OBJECT", "properties": {}}},
]}]

model, data = build_scene(os.environ["MENAGERIE"])
renderer = mujoco.Renderer(model, 480, 640)
sim_lock = asyncio.Lock()                    # never render while physics is stepping

def capture():
    rgb, depth, _ = render_rgbd(renderer, data, "top")
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return buf.getvalue(), depth

def execute_tool(name, args, depth):
    if name == "ack":
        return {"status": "idle"}
    if name == "pick_and_place":
        pick = er_point_to_world(model, data, "top", [args["y"], args["x"]], depth) - [0, 0, 0.02]
        place = er_point_to_world(model, data, "top", [args["place_y"], args["place_x"]], depth) + [0, 0, 0.03]
        ok, msg = pick_and_place(model, data, pick, place)       # add workspace checks here
        return {"status": "success" if ok else "error", "detail": msg}
    return {"status": "error", "detail": f"unknown tool {name}"}

async def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        tools=TOOLS,
        system_instruction=types.Content(parts=[types.Part(text=
            "You control a Franka arm seen from a fixed top camera. Use tools to act. Coordinates are [y, x] 0-1000.")]),
        context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        session_resumption=types.SessionResumptionConfig(handle=None),   # pass the saved handle to resume
    )
    turn_done = asyncio.Event()
    state = {"depth": None, "handle": None}

    async with client.aio.live.connect(model=MODEL, config=config) as session:

        async def receive_loop():
            async for msg in session.receive():
                if msg.session_resumption_update and msg.session_resumption_update.resumable:
                    state["handle"] = msg.session_resumption_update.new_handle
                if msg.go_away:                                   # about 60 s before the connection closes
                    print("server going away in", msg.go_away.time_left)
                if msg.tool_call_cancellation:                    # model cancelled in-flight calls: stop motion
                    print("cancelled:", msg.tool_call_cancellation.ids)
                if msg.server_content:
                    sc = msg.server_content
                    for p in (sc.model_turn.parts if sc.model_turn else []) or []:
                        if p.text:
                            print(p.text, end="", flush=True)
                    if sc.turn_complete:
                        turn_done.set()
                if msg.tool_call:
                    responses = []
                    for call in msg.tool_call.function_calls:
                        async with sim_lock:                      # long physics off the event loop
                            result = await asyncio.to_thread(execute_tool, call.name, call.args, state["depth"])
                        responses.append(types.FunctionResponse(id=call.id, name=call.name, response=result))
                    await session.send_tool_response(function_responses=responses)

        async def heartbeat():                                    # ~1 Hz, never interrupts a turn
            loop = asyncio.get_running_loop()
            while True:
                t0 = loop.time()
                async with sim_lock:
                    jpeg, state["depth"] = capture()
                await session.send_realtime_input(video=types.Blob(data=jpeg, mime_type="image/jpeg"))
                await session.send_realtime_input(text=(
                    "[HEARTBEAT] Goal: every red cube must be in the dark tray. Observe the latest frame. "
                    "If a red cube is outside the tray, call 'pick_and_place'. Otherwise call 'ack'."))
                await turn_done.wait(); turn_done.clear()
                await asyncio.sleep(max(0.0, 1.0 - (loop.time() - t0)))

        await asyncio.gather(receive_loop(), heartbeat())

asyncio.run(main())
```

Other inputs:

```python
# One explicit user turn (image + instruction), ordered in context:
await session.send_client_content(turns=types.Content(role="user", parts=[
    types.Part(inline_data=types.Blob(data=jpeg, mime_type="image/jpeg")),
    types.Part(text="Put the red cube in the tray.")]), turn_complete=True)

# Microphone audio (VAD decides when the user's turn ends):
await session.send_realtime_input(audio=types.Blob(data=pcm16k, mime_type="audio/pcm;rate=16000"))
await session.send_realtime_input(audio_stream_end=True)          # when the mic pauses
```

## Heartbeat rules

- **Wait for `turn_complete` before the next heartbeat.** A new prompt while the model is generating is
  a **barge-in**. It cancels in-flight reasoning **and tool calls**. A blind timer traps the agent in a
  cancellation loop, and it looks frozen.
- **Respect 1 FPS.** Send the latest frame, never a backlog.
- Heartbeats should force an explicit decision: an `ack` tool, `run_instruction(next_step)`, or
  `reset`. The docs' heartbeat text:
  `"[HEARTBEAT] If no task is active, call 'ack' and wait for user input. If a task is active: observe the scene. If the current step is progressing correctly, call 'ack'. If the current step is complete, call 'run_instruction' with the next step. If the overall goal is achieved, call 'reset' and inform the user."`
- While a BLOCKING tool runs, don't send new input. The heartbeat above naturally pauses, because the turn
  isn't complete until after your tool response.

## Long sessions

| Limit | Mitigation |
|---|---|
| WebSocket connection about 10 min | `session_resumption=SessionResumptionConfig(handle=saved)`. Store `new_handle` from `session_resumption_update`, and reconnect on `go_away` or on a drop. The handle is valid for about 2 h on the Developer API. |
| Context: video is about 258 tokens/s and audio about 25 tokens/s (128k window) | `context_window_compression=ContextWindowCompressionConfig(sliding_window=SlidingWindow(), trigger_tokens=...)`. The system instruction is always kept. |
| Need exactly-once message replay | `SessionResumptionConfig(transparent=True)` adds `last_consumed_client_message_index` |

Zero-data-retention apps should not enable session resumption, because it caches context server-side.

## Speech through TTS as a tool

ER outputs text. Expose speech as a tool so "say something" follows the same path as "move the arm".
The same tool can message peer robots:

```python
{"name": "send_message", "behavior": "BLOCKING",
 "description": "Speak a message aloud via TTS, then deliver it to the specified target. Use target='user' to "
                "speak directly to the user, or a peer agent name (e.g., 'duo') to communicate with another robot.",
 "parameters": {"type": "OBJECT", "properties": {
     "target": {"type": "STRING", "description": "Recipient: 'user' or a peer agent name."},
     "message": {"type": "STRING", "description": "The message to speak and deliver."}},
     "required": ["target", "message"]}}
```

Fulfil it with Gemini TTS or any other TTS provider. This keeps voice selection and latency under your control.

## Browser agent with ephemeral tokens

**Server** (never ship the API key). Mint a single-use token constrained to the robotics model:

```ts
import { GoogleGenAI } from '@google/genai';
const server = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const token = await server.authTokens.create({ config: {
  uses: 1,
  expireTime: new Date(Date.now() + 30 * 60_000).toISOString(),          // messages allowed for 30 min
  newSessionExpireTime: new Date(Date.now() + 60_000).toISOString(),     // must connect within 1 min
  liveConnectConstraints: { model: 'gemini-robotics-er-2-streaming-preview',
                            config: { responseModalities: ['TEXT'], sessionResumption: {} } },
}});
// return token.name to the authenticated client
```

**Client** (the Franka demo's browser sim):

```ts
import { GoogleGenAI, Modality, Behavior, Type } from '@google/genai';
const ai = new GoogleGenAI({ apiKey: tokenName, httpOptions: { apiVersion: 'v1alpha' } });  // SDK: ephemeral tokens are v1alpha-only
let turnDone: () => void = () => {};
const session = await ai.live.connect({
  model: 'gemini-robotics-er-2-streaming-preview',
  config: {
    responseModalities: [Modality.TEXT],
    tools: [{ functionDeclarations: [{
      name: 'pick', behavior: Behavior.BLOCKING,
      description: 'Pick the object at [y, x] (0-1000) in the latest frame and drop it in the tray.',
      parameters: { type: Type.OBJECT, properties: { y: { type: Type.INTEGER }, x: { type: Type.INTEGER } }, required: ['y', 'x'] },
    }] }],
  },
  callbacks: {
    onmessage: async (msg: any) => {
      if (msg.serverContent?.turnComplete) turnDone();
      if (msg.toolCall) {
        const functionResponses = [];
        for (const fc of msg.toolCall.functionCalls ?? []) {
          // must use the same camera pose that produced the frame (see RenderSystem.project2DTo3D)
          const hit = sim.renderSys.project2DTo3D(fc.args.x, fc.args.y, capturePos, captureTarget);
          const ok = hit ? await new Promise<boolean>(res => sim.pickupItems([hit.point], [0], () => res(true))) : false;
          functionResponses.push({ id: fc.id, name: fc.name, response: { status: ok ? 'success' : 'error' } });
        }
        session.sendToolResponse({ functionResponses });
      }
    },
    onerror: (e: any) => console.error(e), onclose: (e: any) => console.warn('closed', e.reason),
  },
});

async function heartbeatLoop() {
  while (running) {
    const t0 = performance.now();
    const jpegBase64 = captureJpegBase64();                 // fixed perception camera, ≤ 768 px, quality 0.8
    session.sendRealtimeInput({ video: { data: jpegBase64, mimeType: 'image/jpeg' } });
    session.sendRealtimeInput({ text: '[HEARTBEAT] Pick any red cube outside the tray, else do nothing.' });
    await new Promise<void>(r => (turnDone = r));          // wait for turn_complete
    await new Promise(r => setTimeout(r, Math.max(0, 1000 - (performance.now() - t0))));
  }
}
```

`@google/genai` 2.30 detects keys starting with `auth_tokens/` and connects to the constrained Live
endpoint. Its token support is **v1alpha-only**: without `httpOptions: { apiVersion: 'v1alpha' }` it logs a
warning. Use the same option on the server client if `authTokens.create` fails.

## Pitfalls

| Symptom | Cause / fix |
|---|---|
| Agent "freezes", tools never complete | Heartbeat on a blind timer causes barge-in. Wait for `turn_complete`. |
| Nothing happens when frames stream | Frames alone don't trigger a turn. Add the heartbeat text, or use audio. |
| Session drops after about 2 min with video | No context compression. Enable `context_window_compression`. |
| Drops at about 10 min | Connection lifetime. Use session resumption plus reconnect on `go_away`. |
| `media=` deprecation warnings | Use `send_realtime_input(video=...)`, `audio=` or `text=`. Some older docs still show `media=`. |
| Event loop stalls during motion | Run physics or robot calls with `asyncio.to_thread`, and keep `receive()` responsive |
| Model acts on stale coordinates | Coordinates refer to the most recent frame. Store the depth or camera pose captured with each frame. |
| Structured JSON requested, prose returned | The streaming model has no structured output. Use tools for structure (arguments are typed). |
