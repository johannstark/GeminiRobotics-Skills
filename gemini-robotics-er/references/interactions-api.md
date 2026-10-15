# Interactions API reference for Gemini Robotics-ER 2

The source is <https://ai.google.dev/gemini-api/docs/interactions-overview>, with shapes validated against
`google-genai` 2.31's request models. The Interactions API became GA in June 2026 and is
recommended for new code. `generateContent` is now "legacy", but it is still supported.

## Contents
- [SDK versions](#sdk-versions)
- [Request anatomy](#request-anatomy)
- [Input forms](#input-forms)
- [Structured output](#structured-output)
- [Reading the response](#reading-the-response)
- [Multi-turn state](#multi-turn-state)
- [Thinking](#thinking)
- [Streaming](#streaming)
- [Files API](#files-api)
- [Wrong patterns (seen in the wild)](#wrong-patterns-seen-in-the-wild)
- [Errors](#errors)
- [Migrating the Franka demo](#migrating-the-franka-demo)

## SDK versions

| Language | Package | Minimum | Notes |
|---|---|---|---|
| Python | `google-genai` | `>=2.3.0` (helpers), and `>=2.9.0` as used by the ER 2 notebook | `from google import genai` |
| JS/TS | `@google/genai` | `>=2.3.0` | Request fields are **snake_case** (`generation_config`, `mime_type`, `previous_interaction_id`) |

The legacy `google-generativeai`, `@google/generative-ai`, `google-cloud-aiplatform` and
`@google-cloud/vertexai` packages **do not support Interactions**. If you see a
`TypeError` or "unknown field" error on `interactions`, upgrade the SDK.

## Request anatomy

`client.interactions.create(**body)` takes keyword arguments. The useful fields for ER are:

| Field | Type | Notes |
|---|---|---|
| `model` | str | `"gemini-robotics-er-2-preview"` |
| `input` | str, list of content parts, or list of steps | See [Input forms](#input-forms) |
| `system_instruction` | **str** | A plain string, not a `types.Content` (that type is only for Live). Per-turn. |
| `generation_config` | dict | `thinking_level`, `thinking_summaries` (`auto`/`none`), `temperature`, `top_p`, `seed`, `max_output_tokens`, `stop_sequences`, `tool_choice` (`auto`/`any`/`none`/`validated`). Per-turn. |
| `response_format` | dict or list | `{"type": "text", "mime_type": "application/json", "schema": {...}}` |
| `tools` | list | Functions: `{"type": "function", "name", "description", "parameters"}`. Built-ins: `{"type": "code_execution"}`, `{"type": "google_search"}`, `{"type": "url_context"}`. Per-turn. |
| `previous_interaction_id` | str | Continue server-side state |
| `store` | bool | Default `true`. `false` = stateless (no chaining, no background). |
| `stream` | bool | Server-sent events (see [Streaming](#streaming)) |
| `background` | bool | Long-running jobs. Requires `store=true`. |

## Input forms

```python
input="text only"                                                    # string
input=[{"type": "image", "data": b64, "mime_type": "image/png"},     # flat list of parts
       {"type": "text", "text": prompt}]
input=[{"type": "user_input", "content": [image_part, text_part]}]   # explicit user turn (notebook form)
input=[{"type": "function_result", "call_id": ..., "name": ..., "result": [...]}]  # tool results
```

Part types:

| Part | Fields |
|---|---|
| `image` | `data` (base64 str) **or** `uri` (Files API), `mime_type` (`image/png`, `image/jpeg`, …), optional `resolution`: `low` / `medium` / `high` / `ultra_high` |
| `video` | `data` or `uri`, `mime_type`, optional `processing: {"type": "static", "fps": 2.0, "start_offset": ..., "end_offset": ...}`, optional `resolution` |
| `audio` | `data` or `uri`, `mime_type` |
| `text` | `text` |

- Put media **before** the text prompt. Use one video per request for the best results.
- Raw bytes, PIL images and `types.Part` objects are **not** valid here. Base64-encode the data, or upload it first.
- Images: the notebook resizes to 800 px wide and uses PNG. JPEG at quality 80 is fine for camera frames.
- Uploaded files: `{"type": "image", "uri": f.uri, "mime_type": f.mime_type}` (JS: `f.mimeType`).

## Structured output

```python
from pydantic import BaseModel, Field
class PointItem(BaseModel):
    point: list[int] = Field(description="[y, x] format normalized to 0-1000")
    label: str = Field(description="Identifying name for the object detected")
class PointResponse(BaseModel):
    items: list[PointItem]

interaction = client.interactions.create(
    model=MODEL_ID, input=[...],
    response_format={"type": "text", "mime_type": "application/json",
                     "schema": PointResponse.model_json_schema()},
)
items = PointResponse.model_validate_json(interaction.output_text).items
```

- **Wrap arrays in an object** (`{"items": [...]}`). This is the notebook pattern, and it leaves room for extra fields such as
  `answer` or `reasoning`.
- Supported JSON Schema subset: `string`, `number`, `integer`, `boolean`, `object`, `array` and `null`
  (written as `["string", "null"]`), plus `enum`, `minimum`/`maximum`, `minItems`/`maxItems`,
  `prefixItems`, `required`, `additionalProperties`, `anyOf`, `title` and `description`. Descriptions steer the model.
- JS: pass a plain JSON-schema object. Zod is optional: `z.fromJSONSchema(schema).parse(...)`.
- Structured output combines with `code_execution` and function calling on ER 2.
- The streaming model (Live API) **does not support** structured output. Ask for JSON in the prompt and parse it defensively.

## Reading the response

```python
interaction.id               # pass as previous_interaction_id
interaction.status           # "completed" | "requires_action" (function calls pending) | "incomplete" (hit max_output_tokens) | "failed" | ...
interaction.output_text      # final text, which joins the trailing text blocks. Earlier text split by tool calls is NOT included.
interaction.output_image     # e.g. an annotated image produced by code execution
interaction.steps            # full timeline returned by create (model-generated steps only)
interaction.usage            # token accounting
```

Step types you'll meet with ER:

| `step.type` | Fields |
|---|---|
| `thought` | `signature`, `summary` (if `thinking_summaries: "auto"`) |
| `function_call` | `id`, `name`, `arguments` (dict) |
| `code_execution_call` | `arguments.code` |
| `code_execution_result` | `result` |
| `model_output` | `content` = list of `{type: "text", text}` / `{type: "image", data, mime_type}` |

To get every text block and every image (for example from agentic vision), iterate over `steps`
instead of relying on `output_text`. See `agentic-vision.md`.

## Multi-turn state

```python
r1 = client.interactions.create(model=MODEL_ID, input=[video_part, text_part("Describe each step with timestamps.")])
r2 = client.interactions.create(model=MODEL_ID, previous_interaction_id=r1.id,
                                input=[{"type": "text", "text": "Zoom into 00:15-00:22, per second."}])
```

- Only the conversation (inputs and outputs) carries over. **`tools`, `system_instruction` and
  `generation_config` must be resent every turn.**
- The server manages thought signatures in stateful mode. In stateless mode (`store=false`) you must
  resend the full history, including every `thought` step, unmodified.
- Retention: 55 days on paid, 1 day on free. You can delete with `client.interactions.delete(id)` or in AI Studio logs.

## Thinking

| Level | Use for |
|---|---|
| `minimal` / `low` | Simple pointing or boxes, latency-sensitive loops, orchestration turns (the notebook uses `low`) |
| `medium` | Default trade-off that the robotics docs recommend |
| `high` (ER 2 default) | Counting, payload or affordance logic, multi-view success, moment finding, progress classification |

`thinking_budget` is deprecated for Gemini 3+ and ER 2, and it causes intermittent 400 errors.
`max_output_tokens` includes thinking tokens. A small cap yields `status: "incomplete"` with empty
output. Lower `thinking_level` instead.

## Streaming

```python
for event in client.interactions.create(model=MODEL_ID, input=..., stream=True):
    if event.event_type == "step.delta" and hasattr(event.delta, "text"):
        print(event.delta.text, end="", flush=True)
```

```ts
for await (const event of await client.interactions.create({ model, input, stream: true })) {
  if (event.event_type === 'step.delta' && event.delta && 'text' in event.delta) process.stdout.write(event.delta.text);
}
```

Stream long video or `high`-thinking requests. Without streaming, a long synchronous request can time out
or return 401 after backend retries.

## Files API

```python
f = client.files.upload(file="episode.mp4")          # JS: await client.files.upload({ file, config: { mimeType } })
while f.state.name == "PROCESSING":                  # JS: f.state === 'PROCESSING'
    time.sleep(2); f = client.files.get(name=f.name)
if f.state.name == "FAILED": raise RuntimeError(f.uri)
part = {"type": "video", "uri": f.uri, "mime_type": f.mime_type}
client.files.delete(name=f.name)                     # when done (files also expire automatically)
```

Videos uploaded through the Files API are sampled at **1 FPS** by default, with a timestamp every
second. Override this with `processing.fps`. Supported types include mp4, mpeg, mov, avi, webm, wmv, flv and 3gpp.

## Wrong patterns (seen in the wild)

| Wrong | Right |
|---|---|
| `client.interactions.create(..., config=types.GenerateContentConfig(...))` | `generation_config={...}`, `response_format={...}`, `tools=[...]`, `system_instruction="..."` |
| `input=[pil_image, "prompt"]` | `input=[{"type":"image","data":b64,"mime_type":"image/png"}, {"type":"text","text":"prompt"}]` |
| `response.text`, `response.function_calls` | `interaction.output_text`, `[s for s in interaction.steps if s.type == "function_call"]` |
| `call.args`, `FunctionResponse(...)` on Interactions | `step.arguments`, `{"type":"function_result","call_id":step.id,"name":step.name,"result":[{"type":"text","text":json.dumps(r)}]}` |
| `thinkingConfig: { thinkingBudget: 0 }` | `generation_config: { thinking_level: 'minimal' }` |
| Sending `tools` only on the first turn | Send `tools` on every turn |
| `types.Tool(..., behavior="BLOCKING")` | `behavior` belongs on each **function declaration**, and only in the Live API |
| `session.send(input=..., end_of_turn=True)`, `LiveClientRealtimeInput(media_chunks=...)` | `send_client_content(turns=..., turn_complete=True)`, `send_realtime_input(video=/audio=/text=)` |

## Errors

| Symptom | Cause / fix |
|---|---|
| `403 Forbidden` | The API key is unrestricted. Add restrictions in AI Studio → API keys. |
| `400` with `thinking_budget` | Remove it and use `thinking_level` |
| `400` "input" validation | Wrong part shape (raw bytes or PIL), or a missing `mime_type` |
| Follow-up "forgets" the image | `store=false`, or `previous_interaction_id` not passed |
| Follow-up ignores tools or format | You didn't resend `tools` / `response_format` / `generation_config` |
| Empty `output_text` and `status: "incomplete"` | `max_output_tokens` too small for the thinking. Raise it or lower `thinking_level`. |
| Video request fails | File still `PROCESSING`. Poll until `ACTIVE`. |
| `gs://...` not found | The Developer API can't read GCS. Upload with the Files API. |

## Migrating the Franka demo

`App.tsx::handleErSend` uses `@google/genai@^1.28` and `generateContent`. To migrate:

1. Bump to `"@google/genai": "^2.3.0"` in `package.json` **and** in the `index.html` import map
   (`https://esm.sh/@google/genai@^2.3.0`). AI Studio uses the import map.
2. Replace the call:

```diff
- const config: any = { temperature, responseMimeType: "application/json" };
- if (!enableThinking) config.thinkingConfig = { thinkingBudget: 0 };
- const response = await ai.models.generateContent({
-   model: modelId,
-   contents: { parts: [{ inlineData: { mimeType: 'image/png', data: base64Data } }, { text: textPrompt }] },
-   config,
- });
- const text = response.text;
+ const interaction = await ai.interactions.create({
+   model: modelId,
+   input: [
+     { type: 'image', data: base64Data, mime_type: 'image/png' },
+     { type: 'text', text: textPrompt },
+   ],
+   generation_config: { temperature, thinking_level: enableThinking ? 'medium' : 'minimal' },
+   response_format: { type: 'text', mime_type: 'application/json' },   // or add a schema
+ });
+ const text = interaction.output_text;
```

3. Keep the existing fence-stripping, bracket slicing and dedupe. They are still needed.
   If you add a schema with an `items` wrapper, read `JSON.parse(text).items`.
4. Update `requestLogData` so the log modal shows the new request shape.
5. For production, move the call behind a small server endpoint (for example a Vite middleware or Express)
   and stop defining `process.env.API_KEY` in `vite.config.ts`.
6. `gemini-flash-latest` in the sidebar model picker also works with Interactions on Gemini 3+.
