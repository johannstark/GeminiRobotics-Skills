#!/usr/bin/env python3
"""Helpers for Gemini Robotics-ER 2 with the Google Gen AI SDK 2.x Interactions API and Live API.

Copy this file into your project or import it. Request shapes follow the official ER 2
notebook (https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb)
and https://ai.google.dev/gemini-api/docs/robotics-overview.

    pip install "google-genai>=2.9.0" pydantic pillow opencv-python numpy

Contents
  encode_image()            resize + base64 (the notebook resizes to 800 px wide)
  image_part(), text_part(), user_input()
  POINTS_FORMAT, BOXES_FORMAT, json_format()   response_format dicts for structured output
  ask_json()                one-shot call -> parsed JSON
  parse_er_json()           defensive parser for model JSON (fences, prose, dupes, bad coords)
  yx_to_pixel(), box_center_yx(), relative_to_absolute_yx()
  run_orchestration()       notebook section 4 tool loop with real results, step cap,
                            and single-physical-action enforcement
  upload_and_wait(), video_part(), mmss_to_seconds()   notebook section 5 helpers
  snap_to_detection(), TargetLock  metric target locking and snap helpers
  LiveRobotSession          reusable resilient Live API wrapper with auto-resumption and compression
  check_schema_hygiene()    verifies declared tool schema matches handler argument keys (kwargs or dict)
  lint_prompt_rules()       example prompt lint checking for contradictory pointing rules
  FakeLiveSession           offline test fake yielding types.LiveServerMessage objects
  FakeLiveClient            offline test fake client whose connect() is an asynccontextmanager
  ScriptedNoisyModel        synthetic model injecting Gaussian pointing noise (sigma = 8/1000)
  HiddenOffsetRobot         sim wrapper injecting hidden encoder offsets

Run `python er_tools.py --selftest` to check the offline logic (no API key needed).
"""

from __future__ import annotations

import ast
import asyncio
import base64
import contextlib
import inspect
import io
import json
import math
import re
import textwrap
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Iterable, Sequence

from google.genai import types

MODEL_ID = "gemini-robotics-er-2-preview"
STREAMING_MODEL_ID = "gemini-robotics-er-2-streaming-preview"  # Live API only

# 80/1000 bounds typical ER pointing jitter of ~25 mm at 5 cm hover height
DEFAULT_SNAP_RADIUS_NORM = 80
# 40 mm matches typical manipulated object radius plus grasp tolerance margin
DEFAULT_METRIC_LOCK_RADIUS_MM = 40.0

# Live API reconnection constants
# 3 retries allows transient connection blips to recover without stalling indefinitely
MAX_RESUME_ATTEMPTS = 3
# 1.0 s initial delay gives the server-side socket time to reset after a drop
INITIAL_RETRY_DELAY_S = 1.0
# 2.0x exponential backoff multiplier balances retry speed and server load
BACKOFF_MULTIPLIER = 2.0
# 0.5 s timeout fails closed if no fresh camera frame arrives after robot motion
FRAME_FRESHNESS_TIMEOUT_S = 0.5


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


# ------------------------------------------------------------------ coordinates & metric grounding

def yx_to_pixel(yx, width: int, height: int) -> tuple[float, float]:
    """[y, x] 0-1000 -> (u, v) pixels in the image that was SENT to the model."""
    return yx[1] / 1000 * width, yx[0] / 1000 * height


def box_center_yx(box) -> list[float]:
    ymin, xmin, ymax, xmax = box
    return [(ymin + ymax) / 2, (xmin + xmax) / 2]


def relative_to_absolute_yx(x_rel: float, y_rel: float, origin_yx) -> list[float]:
    """Orchestration example: move(x, y) is relative to an origin given as a normalized [y, x]."""
    return [origin_yx[0] + y_rel, origin_yx[1] + x_rel]


def snap_to_detection(
    er_yx: Sequence[float],
    detections_yx: Sequence[Sequence[float]],
    max_radius: float = DEFAULT_SNAP_RADIUS_NORM,
) -> Sequence[float]:
    """Snap ER 0-1000 normalized point to the closest CV-detected centroid within max_radius.

    If no detection is within max_radius, returns the original ER point.
    """
    if not detections_yx:
        return er_yx
    best_dist = float("inf")
    best_det = er_yx
    ey, ex = er_yx[0], er_yx[1]
    for det in detections_yx:
        dist = math.hypot(det[0] - ey, det[1] - ex)
        if dist < best_dist:
            best_dist = dist
            best_det = det
    if best_dist <= max_radius:
        return best_det
    return er_yx


class TargetLock:
    """Maintains consistent object identity across multiple turns in metric table space (mm).

    Prevents target lock churn caused by ~25 mm ER visual drift between calls.
    """

    def __init__(self, match_radius_mm: float = DEFAULT_METRIC_LOCK_RADIUS_MM):
        if match_radius_mm <= 0:
            raise ValueError(f"match_radius_mm must be positive, got {match_radius_mm}.")
        self.match_radius_mm = match_radius_mm
        self.locked_xy_mm: tuple[float, float] | None = None
        self.locked_canonical_yx: list[float] | None = None

    def lock_or_match(
        self,
        candidate_xy_mm: tuple[float, float],
        candidate_yx: Sequence[float],
    ) -> tuple[tuple[float, float], bool]:
        """Match candidate against locked table-space position or establish lock on first call.

        Returns (canonical_xy_mm, is_matched).
        """
        if self.locked_xy_mm is None:
            self.locked_xy_mm = candidate_xy_mm
            self.locked_canonical_yx = list(candidate_yx)
            return self.locked_xy_mm, True

        dist = math.hypot(
            candidate_xy_mm[0] - self.locked_xy_mm[0],
            candidate_xy_mm[1] - self.locked_xy_mm[1],
        )
        if dist <= self.match_radius_mm:
            return self.locked_xy_mm, True

        # Candidate is outside tolerance: treat as a distinct new target
        self.locked_xy_mm = candidate_xy_mm
        self.locked_canonical_yx = list(candidate_yx)
        return self.locked_xy_mm, False


# ------------------------------------------------------------------ orchestration (notebook section 4)

def run_orchestration(
    client,
    first_input: list,
    tools: list[dict],
    execute: Callable[[str, dict], dict],
    *,
    max_steps: int = 15,
    model: str = MODEL_ID,
    thinking_level: str = "low",
    system_instruction: str | None = None,
    on_step: Callable[[str, dict, dict], None] | None = None,
    physical_tools: set[str] | None = None,
) -> dict:
    """ER plans, your code acts. Returns {"done": bool, "summary": str, "calls": [...], "interaction_id": str}.

    execute(name, args) must validate + run the action and return a JSON-serialisable result.
    Include {"status": "error", ...} (or raise) on failure so the model can re-plan.
    Exceptions are returned to the model with is_error=True.
    tools, generation_config and system_instruction are per-interaction, so they are resent each turn.

    If physical_tools is supplied, only the first physical tool call in a multi-call turn is executed;
    subsequent physical calls in that turn receive an error requiring observation before re-issuing.
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
        executed_physical_this_turn = False

        for step in tool_calls:
            args = dict(step.arguments or {})
            is_physical = physical_tools is not None and step.name in physical_tools

            # Enforce single physical action per turn rule
            if is_physical and executed_physical_this_turn:
                result = {
                    "status": "error",
                    "reason": "one physical action per turn; re-issue after observing",
                }
                is_error = True
            else:
                try:
                    result, is_error = execute(step.name, args), False
                    if is_physical:
                        executed_physical_this_turn = True
                except Exception as e:
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


# ------------------------------------------------------------------ schema & prompt hygiene

class _DictKeyExtractor(ast.NodeVisitor):
    """AST visitor that extracts string keys accessed on a specific dictionary variable."""

    def __init__(self, dict_var_name: str):
        self.dict_var_name = dict_var_name
        self.accessed_keys: set[str] = set()

    def visit_Subscript(self, node: ast.Subscript):
        # args["key"]
        if isinstance(node.value, ast.Name) and node.value.id == self.dict_var_name:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                self.accessed_keys.add(node.slice.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # args.get("key") or args.pop("key")
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == self.dict_var_name and node.func.attr in ("get", "pop"):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    self.accessed_keys.add(node.args[0].value)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        # "key" in args
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, ast.Name) and comp.id == self.dict_var_name:
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    self.accessed_keys.add(node.left.value)
        self.generic_visit(node)


def _extract_dict_handler_keys(handler: Callable, arg_name: str) -> set[str] | None:
    """Attempts to inspect handler source AST to find keys accessed from arg_name."""
    try:
        source = textwrap.dedent(inspect.getsource(handler))
        tree = ast.parse(source)
        extractor = _DictKeyExtractor(arg_name)
        extractor.visit(tree)
        return extractor.accessed_keys
    except Exception:
        return None


def check_schema_hygiene(
    tool_declarations: Sequence[dict],
    handlers: dict[str, Callable],
) -> None:
    """Verify tool declaration schemas match handler implementations.

    Supports both keyword-parameter handlers (`def h(y, x): ...`) and dict-style
    handlers (`def h(args): args.get('k')` or `args['k']`).

    Raises AssertionError if:
    - A handler expects or reads a parameter not declared in its schema.
    - A declared required parameter is never accepted or read by the handler.
    """
    for decl in tool_declarations:
        # Handles both Interactions flat function dicts and Live function_declarations dicts
        funcs = decl.get("function_declarations", [decl] if decl.get("type") == "function" or "name" in decl else [])
        for fn_decl in funcs:
            name = fn_decl.get("name")
            if not name or name not in handlers:
                continue
            handler = handlers[name]
            sig = inspect.signature(handler)
            params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
            handler_params = set(p.name for p in params)
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

            params_schema = fn_decl.get("parameters", {})
            declared_props = set(params_schema.get("properties", {}).keys())
            required_props = set(params_schema.get("required", []))

            # Detect dict-style handler: single argument whose name is not declared as a property
            is_dict_handler = (
                len(params) == 1
                and params[0].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
                and params[0].name not in declared_props
            )

            if is_dict_handler:
                dict_arg_name = params[0].name
                read_keys = _extract_dict_handler_keys(handler, dict_arg_name)
                if read_keys is None:
                    raise AssertionError(
                        f"cannot inspect handler source for '{name}'; use keyword parameters or check it manually"
                    )
                for k in read_keys:
                    if k not in declared_props:
                        raise AssertionError(
                            f"Handler '{name}' reads undeclared argument '{k}' from '{dict_arg_name}'. "
                            f"Add it to the tool declaration properties."
                        )
                for req in required_props:
                    if req not in read_keys:
                        raise AssertionError(
                            f"Tool declaration '{name}' requires '{req}', but handler '{name}' does not read it from '{dict_arg_name}'."
                        )
                continue

            # Handlers without **kwargs cannot accept undeclared arguments
            if not has_var_kwargs:
                for param in handler_params:
                    if param not in declared_props:
                        raise AssertionError(
                            f"Handler '{name}' reads undeclared argument '{param}'. "
                            f"Add it to the tool declaration properties."
                        )

            # Every required property must be an explicit parameter or consumed via **kwargs
            for req in required_props:
                if req not in handler_params and not has_var_kwargs:
                    raise AssertionError(
                        f"Tool declaration '{name}' requires '{req}', but handler does not accept it."
                    )


def lint_prompt_rules(prompt_text: str) -> None:
    """Example prompt lint checking for conflicting pointing contracts or duplicate ambiguous rules.

    This illustrates how controller test harnesses can statically lint application-specific
    system prompts (e.g. flagging contradictory instructions like pointing at the visual center
    versus the bottom contact footprint). Projects should adapt or extend this pattern for
    their own prompt contracts.
    """
    text_lower = prompt_text.lower()
    has_center_rule = "point at the object itself" in text_lower or "point at the visual center" in text_lower
    has_contact_rule = "bottom contact" in text_lower or "footprint center" in text_lower

    if has_center_rule and has_contact_rule:
        raise AssertionError(
            "Conflicting pointing contracts in prompt: prompt specifies both visual center and "
            "bottom contact point. Enforce exactly one pointing contract."
        )


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


# ------------------------------------------------------------------ resilient Live API wrapper

class LiveRobotSession:
    """Resilient wrapper for the Live API with automatic session resumption and context compression.

    - Uses `async with client.aio.live.connect(...)` managed via contextlib.AsyncExitStack.
    - Closes old session cleanly on reconnect.
    - Stores and updates resumption handle across reconnections.
    - Keeps retrying with exponential backoff when reconnection itself fails.
    - Retries only transient errors (connection closed, go_away, timeouts); raises everything else after stopping the robot.
    - Forces sliding-window compression on video streaming sessions (~258 tokens/s).
    - Serialises outgoing message writes with an asyncio lock.
    - Stops the robot safely via callback whenever the connection state is uncertain.
    """

    def __init__(
        self,
        client: Any,
        model: str = STREAMING_MODEL_ID,
        config: types.LiveConnectConfig | None = None,
        stop_robot_callback: Callable[[], None] | None = None,
        video_mode: bool = True,
    ):
        self.client = client
        self.model = model
        self.stop_robot_callback = stop_robot_callback
        self.video_mode = video_mode
        self.write_lock = asyncio.Lock()
        self.resumption_handle: str | None = None
        self.raw_session: Any = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._active = False

        # Build resilient default configuration
        if config is None:
            config = types.LiveConnectConfig(response_modalities=["TEXT"])

        # Mandatory sliding-window compression in video mode to prevent hitting ~2 min limit
        if video_mode and config.context_window_compression is None:
            config.context_window_compression = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            )

        if config.session_resumption is None:
            config.session_resumption = types.SessionResumptionConfig(handle=None)

        self.config = config

    def _is_transient_error(self, exc: BaseException) -> bool:
        """Determines if an error is transient (network drop, timeout, go_away)."""
        if isinstance(
            exc,
            (ConnectionError, ConnectionResetError, BrokenPipeError, asyncio.TimeoutError, TimeoutError),
        ):
            return True
        msg = str(exc).lower()
        transient_keywords = (
            "connection",
            "closed",
            "reset",
            "timeout",
            "timed out",
            "unavailable",
            "temporarily",
            "go_away",
            "1006",
            "disconnect",
            "econnreset",
            "socket",
        )
        return any(k in msg for k in transient_keywords)

    async def _safe_stop(self):
        """Invoke safety stop callback when connection state drops or becomes uncertain."""
        if self.stop_robot_callback:
            try:
                if inspect.iscoroutinefunction(self.stop_robot_callback):
                    await self.stop_robot_callback()
                else:
                    self.stop_robot_callback()
            except Exception:
                pass

    async def _establish_connection(self):
        """Establish or re-establish live connection, closing any existing session."""
        await self._exit_stack.aclose()
        self._exit_stack = contextlib.AsyncExitStack()
        self.raw_session = None

        if self.resumption_handle:
            self.config.session_resumption.handle = self.resumption_handle

        cm = self.client.aio.live.connect(model=self.model, config=self.config)
        self.raw_session = await self._exit_stack.enter_async_context(cm)

    async def connect(self):
        """Establish initial live connection."""
        self._active = True
        await self._establish_connection()
        return self

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def send_realtime_input(self, **kwargs) -> None:
        """Send realtime media chunks (video, text, audio) serialised under write lock."""
        async with self.write_lock:
            if self.raw_session:
                await self.raw_session.send_realtime_input(**kwargs)

    async def send_tool_response(self, function_responses: list) -> None:
        """Send tool responses serialised under write lock."""
        async with self.write_lock:
            if self.raw_session:
                await self.raw_session.send_tool_response(function_responses=function_responses)

    async def _reconnect_with_retry(self):
        """Keep retrying when the reconnect itself fails, retrying only transient errors."""
        retries = 0
        delay = INITIAL_RETRY_DELAY_S

        while self._active:
            retries += 1
            if retries > MAX_RESUME_ATTEMPTS:
                raise ConnectionError(
                    f"Live session failed to reconnect after {MAX_RESUME_ATTEMPTS} attempts."
                )
            await asyncio.sleep(delay)
            delay = min(delay * BACKOFF_MULTIPLIER, 10.0)

            try:
                async with self.write_lock:
                    await self._establish_connection()
                return
            except Exception as exc:
                await self._safe_stop()
                if not self._is_transient_error(exc):
                    raise
                # Transient error: keep retrying

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        """Receive stream yielding server messages with transparent reconnection on drops."""
        while self._active:
            try:
                async for msg in self.raw_session.receive():
                    # Record updated resumption handle
                    if msg.session_resumption_update and msg.session_resumption_update.resumable:
                        self.resumption_handle = msg.session_resumption_update.new_handle

                    yield msg

                    # Server go_away warning: reconnect gracefully before disconnect
                    if getattr(msg, "go_away", None) is not None:
                        await self._safe_stop()
                        await self._reconnect_with_retry()
                        break
                else:
                    # Clean termination from server
                    break

            except Exception as exc:
                await self._safe_stop()
                if not self._is_transient_error(exc):
                    await self._exit_stack.aclose()
                    self.raw_session = None
                    raise

                await self._reconnect_with_retry()

    async def close(self):
        """Gracefully close session and release resources."""
        self._active = False
        await self._safe_stop()
        await self._exit_stack.aclose()
        self.raw_session = None


# ------------------------------------------------------------------ offline testing fakes

class FakeLiveSession:
    """Async context manager fake yielding predefined types.LiveServerMessage instances."""

    def __init__(self, messages: Sequence[Any]):
        self.messages = list(messages)
        self.sent_realtime: list[dict] = []
        self.sent_responses: list[list] = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.closed = True

    async def send_realtime_input(self, **kwargs):
        self.sent_realtime.append(kwargs)

    async def send_tool_response(self, function_responses: list):
        self.sent_responses.append(function_responses)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        for item in self.messages:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def close(self):
        self.closed = True


class FakeLiveClient:
    """Fake client whose aio.live.connect() is an async context manager."""

    def __init__(self, sessions: Sequence[Any]):
        self.sessions = list(sessions)
        self.connect_calls: list[dict[str, Any]] = []
        self.aio = SimpleNamespace(
            live=SimpleNamespace(connect=self._connect)
        )

    @contextlib.asynccontextmanager
    async def _connect(self, model: str, config: Any):
        handle = getattr(getattr(config, "session_resumption", None), "handle", None)
        self.connect_calls.append({"model": model, "config": config, "handle": handle})
        if not self.sessions:
            raise ConnectionError("No more fake sessions queued in FakeLiveClient")
        item = self.sessions.pop(0)
        if isinstance(item, BaseException):
            raise item
        try:
            item.closed = False
            yield item
        finally:
            item.closed = True


def _selftest_dict_handler(args):
    """Module-level dict handler for AST inspection selftest."""
    return args.get("label")


def _selftest_dict_handler_bad(args):
    """Module-level dict handler reading undeclared key for AST inspection selftest."""
    return args.get("undeclared_key")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def send_realtime_input(self, **kwargs):
        self.sent_realtime.append(kwargs)

    async def send_tool_response(self, function_responses: list):
        self.sent_responses.append(function_responses)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        for msg in self.messages:
            yield msg

    async def close(self):
        pass


class ScriptedNoisyModel:
    """Simulates ER pointing by adding Gaussian noise (sigma = 8/1000) to ground-truth points."""

    def __init__(self, noise_sigma: float = 0.008, seed: int = 42):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.noise_sigma = noise_sigma

    def propose_point(self, true_yx_norm: Sequence[float]) -> list[int]:
        """Add noise and clamp to 0-1000 integer grid."""
        noisy_y = round(true_yx_norm[0] + self.rng.normal(0, self.noise_sigma * 1000))
        noisy_x = round(true_yx_norm[1] + self.rng.normal(0, self.noise_sigma * 1000))
        return [max(0, min(1000, noisy_y)), max(0, min(1000, noisy_x))]


class HiddenOffsetRobot:
    """Robot sim wrapper that injects an unmodeled encoder offset (e.g. 1.5 deg pan error)."""

    def __init__(self, real_robot, pan_offset_rad: float = math.radians(1.5)):
        self.real_robot = real_robot
        self.pan_offset_rad = pan_offset_rad

    def move_pan(self, commanded_rad: float):
        """Move with hardware encoder offset that FK ignores but external camera sees."""
        actual_rad = commanded_rad + self.pan_offset_rad
        return self.real_robot.move_pan(actual_rad)


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

    # Test snap_to_detection
    snapped = snap_to_detection([310, 510], [[300, 500], [800, 800]])
    assert snapped == [300, 500], f"Expected [300, 500], got {snapped}"
    out_of_range = snap_to_detection([100, 100], [[300, 500]], max_radius=80)
    assert out_of_range == [100, 100]

    # Test TargetLock
    lock = TargetLock(match_radius_mm=40.0)
    pos1, matched1 = lock.lock_or_match((200.0, 100.0), [300, 500])
    assert matched1 and pos1 == (200.0, 100.0)
    # Drift within 40 mm matches locked identity
    pos2, matched2 = lock.lock_or_match((215.0, 110.0), [312, 510])
    assert matched2 and pos2 == (200.0, 100.0)
    # Distant point creates new lock
    pos3, matched3 = lock.lock_or_match((350.0, 250.0), [550, 700])
    assert not matched3 and pos3 == (350.0, 250.0)

    # Test schema hygiene
    tools_def = [{
        "type": "function", "name": "pick",
        "parameters": {"type": "object", "properties": {"y": {"type": "int"}, "x": {"type": "int"}}, "required": ["y", "x"]}
    }]
    def valid_handler(y: int, x: int): return {"status": "ok"}
    def bad_handler_reads_undeclared(y: int, x: int, label: str): return {"status": "ok"}

    check_schema_hygiene(tools_def, {"pick": valid_handler})
    try:
        check_schema_hygiene(tools_def, {"pick": bad_handler_reads_undeclared})
        assert False, "Should have caught undeclared argument 'label'"
    except AssertionError as e:
        assert "reads undeclared argument 'label'" in str(e)

    # Test dict-style schema hygiene: def h(args): args.get("label")
    tools_dict_def = [{
        "type": "function", "name": "tag",
        "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
    }]
    check_schema_hygiene(tools_dict_def, {"tag": _selftest_dict_handler})
    try:
        check_schema_hygiene(tools_dict_def, {"tag": _selftest_dict_handler_bad})
        assert False, "Should have caught undeclared argument 'undeclared_key' in dict handler"
    except AssertionError as e:
        assert "reads undeclared argument 'undeclared_key'" in str(e)

    # Test dict-style handler whose source is unavailable (built via exec)
    scope: dict[str, Any] = {}
    exec('def exec_handler(args): return args.get("label")', scope)
    try:
        check_schema_hygiene(tools_dict_def, {"tag": scope["exec_handler"]})
        assert False, "Should have raised when handler source cannot be inspected"
    except AssertionError as e:
        assert "cannot inspect handler source for 'tag'" in str(e)

    # Test prompt linting
    clean_prompt = "Find the cube. Points are normalized [y, x] to 0-1000. Point at the bottom contact point."
    lint_prompt_rules(clean_prompt)
    contradictory_prompt = "Point at the object itself. ALWAYS point at the bottom contact point."
    try:
        lint_prompt_rules(contradictory_prompt)
        assert False, "Should have caught contradictory pointing contract"
    except AssertionError as e:
        assert "Conflicting pointing contracts" in str(e)

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
                            tools=[{"type": "function", "name": "move"}], execute=execute,
                            physical_tools={"move"})
    assert res["done"] and res["summary"] == "Done." and len(res["calls"]) == 3
    assert sent[1]["previous_interaction_id"] == "int_0" and sent[1]["tools"] and sent[1]["generation_config"]
    assert [r["call_id"] for r in sent[1]["input"]] == ["c1", "c2"]
    assert sent[2]["input"][0]["is_error"] is True

    # Test single physical action enforcement in run_orchestration
    double_physical_script = [
        [NS(type="function_call", name="move", id="m1", arguments={"x": 100, "y": 100}),
         NS(type="function_call", name="move", id="m2", arguments={"x": 200, "y": 200})],
        [NS(type="model_output")],
    ]
    sent_double = []
    class FakeDoubleInteractions:
        def create(self, **kw):
            sent_double.append(kw)
            i = len(sent_double) - 1
            return NS(id=f"dbl_{i}", steps=double_physical_script[i], output_text="Stopped." if i == 1 else None)

    res_double = run_orchestration(
        NS(interactions=FakeDoubleInteractions()),
        user_input(text_part("go")),
        tools=[{"type": "function", "name": "move"}],
        execute=execute,
        physical_tools={"move"}
    )
    assert res_double["calls"][0]["result"]["status"] == "success"
    assert res_double["calls"][1]["result"]["status"] == "error"
    assert "one physical action per turn" in res_double["calls"][1]["result"]["reason"]

    # Test ScriptedNoisyModel
    noisy_model = ScriptedNoisyModel(noise_sigma=0.008, seed=42)
    p = noisy_model.propose_point([500, 500])
    assert abs(p[0] - 500) < 30 and abs(p[1] - 500) < 30

    # Test LiveRobotSession against FakeLiveClient (drop mid-stream, go_away, saved handle, non-retryable error)
    async def _test_fake_live():
        stopped_count = 0
        def stop_robot():
            nonlocal stopped_count
            stopped_count += 1

        # 1. Drop mid-stream & resuming with saved handle
        m1 = types.LiveServerMessage(
            session_resumption_update=types.LiveServerSessionResumptionUpdate(new_handle="handle_drop_1", resumable=True),
            server_content=types.LiveServerContent(turn_complete=False, model_turn=types.Content(parts=[types.Part(text="part1")]))
        )
        m2 = types.LiveServerMessage(
            server_content=types.LiveServerContent(turn_complete=True, model_turn=types.Content(parts=[types.Part(text="part2")]))
        )
        s1 = FakeLiveSession([m1, ConnectionResetError("socket dropped mid-stream")])
        s2 = FakeLiveSession([m2])
        client1 = FakeLiveClient([s1, s2])

        wrapper1 = LiveRobotSession(client1, stop_robot_callback=stop_robot)
        await wrapper1.connect()
        received1 = []
        async for msg in wrapper1.receive():
            received1.append(msg)
        await wrapper1.close()

        assert len(received1) == 2
        assert s1.closed is True
        assert s2.closed is True
        assert client1.connect_calls[1]["handle"] == "handle_drop_1"
        assert stopped_count >= 1

        # 2. Server go_away warning with graceful resumption
        m_gw = types.LiveServerMessage(
            go_away=types.LiveServerGoAway(),
            session_resumption_update=types.LiveServerSessionResumptionUpdate(new_handle="handle_gw_2", resumable=True)
        )
        m_after = types.LiveServerMessage(
            server_content=types.LiveServerContent(turn_complete=True, model_turn=types.Content(parts=[types.Part(text="after_gw")]))
        )
        s_gw1 = FakeLiveSession([m_gw])
        s_gw2 = FakeLiveSession([m_after])
        client2 = FakeLiveClient([s_gw1, s_gw2])

        wrapper2 = LiveRobotSession(client2, stop_robot_callback=stop_robot)
        await wrapper2.connect()
        received2 = []
        async for msg in wrapper2.receive():
            received2.append(msg)
        await wrapper2.close()

        assert len(received2) == 2
        assert s_gw1.closed is True
        assert client2.connect_calls[1]["handle"] == "handle_gw_2"

        # 3. Transient failure on reconnect before succeeding
        s_fail = FakeLiveSession([ConnectionResetError("drop")])
        s_ok = FakeLiveSession([m2])
        client3 = FakeLiveClient([s_fail, ConnectionError("transient reconnect gateway error"), s_ok])
        wrapper3 = LiveRobotSession(client3, stop_robot_callback=stop_robot)
        await wrapper3.connect()
        received3 = []
        async for msg in wrapper3.receive():
            received3.append(msg)
        await wrapper3.close()
        assert len(received3) == 1
        assert len(client3.connect_calls) == 3

        # 4. Non-retryable error: stops robot and raises immediately without reconnecting
        s_bad = FakeLiveSession([ValueError("Invalid API key / parameters")])
        client4 = FakeLiveClient([s_bad])
        prev_stops = stopped_count
        wrapper4 = LiveRobotSession(client4, stop_robot_callback=stop_robot)
        await wrapper4.connect()
        try:
            async for _ in wrapper4.receive():
                pass
            assert False, "Should have raised ValueError on non-retryable error"
        except ValueError as err:
            assert "Invalid API key" in str(err)
        assert len(client4.connect_calls) == 1
        assert s_bad.closed is True
        assert stopped_count > prev_stops

    asyncio.run(_test_fake_live())

    print("er_tools selftest: OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
