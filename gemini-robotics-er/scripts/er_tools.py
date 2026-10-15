#!/usr/bin/env python3
"""Helpers for Gemini Robotics-ER 2 with the Google Gen AI SDK 2.x Interactions API.

Copy this file into your project or import it. Request shapes follow the official ER 2
notebook (https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb) and https://ai.google.dev/gemini-api/docs/robotics-overview.

    pip install "google-genai>=2.9.0" pydantic pillow

Contents
  encode_image()            resize + base64 (the notebook resizes to 800 px wide)
  image_part(), text_part(), user_input()
  POINTS_FORMAT, BOXES_FORMAT, json_format()   response_format dicts for structured output
  ask_json()                one-shot call -> parsed JSON
  parse_er_json()           defensive parser for model JSON (fences, prose, dupes, bad coords)
  yx_to_pixel(), box_center_yx(), relative_to_absolute_yx()
  run_orchestration()       notebook section 4 tool loop with real results and a step cap
  upload_and_wait(), video_part(), mmss_to_seconds()   notebook section 5 helpers

Run `python er_tools.py --selftest` to check the offline logic (no API key needed).
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any, Callable, Iterable

MODEL_ID = "gemini-robotics-er-2-preview"
STREAMING_MODEL_ID = "gemini-robotics-er-2-streaming-preview"  # Live API only


# ------------------------------------------------------------------ inputs

def encode_image(img, width: int = 800, fmt: str = "PNG") -> tuple[str, str]:
    """PIL image or HxWx3 uint8 array -> (base64, mime). Keeps aspect ratio."""
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if width and img.width != width:
        img = img.resize((width, round(width * img.height / img.width)), Image.Resampling.LANCZOS)
    if fmt.upper() == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode(), f"image/{fmt.lower()}"


def image_part(b64: str, mime: str = "image/png", resolution: str | None = None) -> dict:
    part = {"type": "image", "data": b64, "mime_type": mime}
    if resolution:                     # "low" | "medium" | "high" | "ultra_high"
        part["resolution"] = resolution
    return part


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def user_input(*parts: dict) -> list[dict]:
    """Wrap content parts as one user turn (the notebook's form)."""
    return [{"type": "user_input", "content": list(parts)}]


# ------------------------------------------------------------------ structured output

def json_format(schema: dict) -> dict:
    return {"type": "text", "mime_type": "application/json", "schema": schema}


_POINT_ITEM = {"type": "object", "properties": {
    "point": {"type": "array", "items": {"type": "integer"}, "description": "[y, x] normalized to 0-1000"},
    "label": {"type": "string"}}, "required": ["point", "label"]}
_BOX_ITEM = {"type": "object", "properties": {
    "box_2d": {"type": "array", "items": {"type": "integer"},
               "description": "[ymin, xmin, ymax, xmax] normalized to 0-1000"},
    "label": {"type": "string"}}, "required": ["box_2d", "label"]}

POINTS_FORMAT = json_format({"type": "object", "properties": {"items": {"type": "array", "items": _POINT_ITEM}},
                             "required": ["items"]})
BOXES_FORMAT = json_format({"type": "object", "properties": {"items": {"type": "array", "items": _BOX_ITEM}},
                            "required": ["items"]})


def ask_json(client, prompt: str, *media: dict, response_format: dict | None = POINTS_FORMAT,
             thinking_level: str = "medium", model: str = MODEL_ID, **kwargs) -> Any:
    """One-shot ER call. Returns parsed JSON (dict/list) or raises ValueError."""
    interaction = client.interactions.create(
        model=model,
        input=user_input(*media, text_part(prompt)),
        generation_config={"thinking_level": thinking_level},
        **({"response_format": response_format} if response_format else {}),
        **kwargs,
    )
    return json.loads(_strip_to_json(interaction.output_text or ""))


# ------------------------------------------------------------------ parsing

def _strip_to_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON in model output: {text[:120]!r}")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start:end + 1]


def parse_er_json(text: str, key: str | None = None, max_items: int = 50) -> list[dict]:
    """Parse points/boxes from ER output, tolerant of fences, prose, {"items": [...]} or bare lists.

    Drops exact duplicates and items whose coordinates are missing or outside 0-1000.
    key: "point" or "box_2d" to require that field (auto-detected when None).
    """
    try:
        data = json.loads(_strip_to_json(text))
    except (ValueError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = data.get("items", [data])
    out, seen = [], set()
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        k = key or ("point" if "point" in item else "box_2d" if "box_2d" in item else None)
        coords = item.get(k) if k else None
        n = 2 if k == "point" else 4
        if not (isinstance(coords, list) and len(coords) == n
                and all(isinstance(c, (int, float)) and 0 <= c <= 1000 for c in coords)):
            continue
        if k == "box_2d" and not (coords[0] < coords[2] and coords[1] < coords[3]):
            continue
        sig = json.dumps(item, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


# ------------------------------------------------------------------ coordinates

def yx_to_pixel(yx, width: int, height: int) -> tuple[float, float]:
    """[y, x] 0-1000 -> (u, v) pixels in the image that was SENT to the model."""
    return yx[1] / 1000 * width, yx[0] / 1000 * height


def box_center_yx(box) -> list[float]:
    ymin, xmin, ymax, xmax = box
    return [(ymin + ymax) / 2, (xmin + xmax) / 2]


def relative_to_absolute_yx(x_rel: float, y_rel: float, origin_yx) -> list[float]:
    """Orchestration example: move(x, y) is relative to an origin given as a normalized [y, x]."""
    return [origin_yx[0] + y_rel, origin_yx[1] + x_rel]


# ------------------------------------------------------------------ orchestration (notebook section 4)

def run_orchestration(client, first_input: list, tools: list[dict],
                      execute: Callable[[str, dict], dict], *, max_steps: int = 15,
                      model: str = MODEL_ID, thinking_level: str = "low",
                      system_instruction: str | None = None,
                      on_step: Callable[[str, dict, dict], None] | None = None) -> dict:
    """ER plans, your code acts. Returns {"done": bool, "summary": str, "calls": [...], "interaction_id": str}.

    execute(name, args) must validate + run the action and return a JSON-serialisable result.
    Include {"status": "error", ...} (or raise) on failure so the model can re-plan. Never
    report blind success. Exceptions are returned to the model with is_error=True.
    tools, generation_config and system_instruction are per-interaction, so they are resent each turn.
    """
    common = {"model": model, "tools": tools, "generation_config": {"thinking_level": thinking_level}}
    if system_instruction:
        common["system_instruction"] = system_instruction
    interaction = client.interactions.create(input=first_input, **common)
    calls: list[dict] = []
    for _ in range(max_steps):
        tool_calls = [s for s in (interaction.steps or []) if s.type == "function_call"]
        if not tool_calls:
            return {"done": True, "summary": interaction.output_text or "", "calls": calls,
                    "interaction_id": interaction.id}
        results = []
        for step in tool_calls:
            args = dict(step.arguments or {})
            try:
                result, is_error = execute(step.name, args), False
            except Exception as e:  # report, don't crash: the model can recover
                result, is_error = {"status": "error", "reason": str(e)}, True
            if isinstance(result, dict) and result.get("status") == "error":
                is_error = True
            calls.append({"name": step.name, "args": args, "result": result})
            if on_step:
                on_step(step.name, args, result)
            fr = {"type": "function_result", "name": step.name, "call_id": step.id,
                  "result": [{"type": "text", "text": json.dumps(result)}]}
            if is_error:
                fr["is_error"] = True
            results.append(fr)
        interaction = client.interactions.create(previous_interaction_id=interaction.id, input=results, **common)
    return {"done": False, "summary": f"step limit {max_steps} reached", "calls": calls,
            "interaction_id": interaction.id}


# ------------------------------------------------------------------ video (notebook section 5)

def upload_and_wait(client, path: str, poll_s: float = 2.0, timeout_s: float = 300.0):
    """Upload with the Files API and block until the file is ACTIVE."""
    f = client.files.upload(file=path)
    t0 = time.time()
    while f.state.name == "PROCESSING":
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{path} still PROCESSING after {timeout_s}s")
        time.sleep(poll_s)
        f = client.files.get(name=f.name)
    if f.state.name == "FAILED":
        raise RuntimeError(f"processing failed: {f.uri}")
    return f


def video_part(file, fps: float | None = None) -> dict:
    """Reference an uploaded video. fps > 1 helps fast motion (default sampling is 1 FPS)."""
    part = {"type": "video", "uri": file.uri, "mime_type": file.mime_type}
    if fps:
        part["processing"] = {"type": "static", "fps": fps}
    return part


def mmss_to_seconds(ts: str) -> float | None:
    """'00:09' -> 9.0, '1:02:03' -> 3723.0, '-1' / '' / None -> None."""
    if ts is None or str(ts).strip() in ("", "-1", "null"):
        return None
    secs = 0.0
    for p in str(ts).strip().split(":"):
        secs = secs * 60 + float(p)
    return secs


# ------------------------------------------------------------------ self test (offline)

def _selftest() -> None:
    from types import SimpleNamespace as NS

    assert parse_er_json('```json\n[{"point": [10, 20], "label": "a"}, {"point": [10, 20], "label": "a"}]\n```') \
        == [{"point": [10, 20], "label": "a"}]
    assert parse_er_json('Sure! {"items": [{"point": [5, 1200], "label": "bad"}, {"point": [5, 6], "label": "ok"}]}') \
        == [{"point": [5, 6], "label": "ok"}]
    assert parse_er_json('[{"box_2d": [100, 100, 50, 200], "label": "inverted"}]') == []
    assert parse_er_json("no json here") == []
    assert box_center_yx([100, 200, 300, 400]) == [200, 300]
    assert yx_to_pixel([500, 250], 800, 600) == (200.0, 300.0)
    assert relative_to_absolute_yx(160, 435, [300, 500]) == [735, 660]
    assert mmss_to_seconds("00:09") == 9 and mmss_to_seconds("-1") is None and mmss_to_seconds("1:02:03") == 3723

    # Fake Interactions client: plans open -> move -> (fails once) -> summary.
    script = [
        [NS(type="function_call", name="setGripperState", id="c1", arguments={"opened": True}),
         NS(type="function_call", name="move", id="c2", arguments={"x": 160, "y": 435, "high": True})],
        [NS(type="function_call", name="move", id="c3", arguments={"x": 9999, "y": 0, "high": False})],
        [NS(type="thought"), NS(type="model_output")],
    ]
    sent = []

    class FakeInteractions:
        def create(self, **kw):
            sent.append(kw)
            i = len(sent) - 1
            return NS(id=f"int_{i}", steps=script[i], output_text="Done." if i == 2 else None)

    def execute(name, args):
        if name == "move" and abs(args["x"]) > 1000:
            raise ValueError("target outside workspace")
        return {"status": "success"}

    res = run_orchestration(NS(interactions=FakeInteractions()), user_input(text_part("go")),
                            tools=[{"type": "function", "name": "move"}], execute=execute)
    assert res["done"] and res["summary"] == "Done." and len(res["calls"]) == 3
    assert sent[1]["previous_interaction_id"] == "int_0" and sent[1]["tools"] and sent[1]["generation_config"]
    assert [r["call_id"] for r in sent[1]["input"]] == ["c1", "c2"]
    assert sent[2]["input"][0]["is_error"] is True
    print("er_tools selftest: OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
