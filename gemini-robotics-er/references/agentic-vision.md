# Agentic vision: code execution (notebook section 3)

With `tools=[{"type": "code_execution"}]`, ER 2 writes and runs Python mid-generation (OpenCV,
NumPy, matplotlib) to crop, zoom, rotate, measure, and draw on the image. Use it for fine detail
such as gauges, liquid levels, chip markings, small labels, precise boxes in clutter, and annotated images.
It only works on `gemini-robotics-er-2-preview`, not on the streaming model.

Inside the sandbox the image you sent is a file named `input_file_0.png`. Files the model saves can
come back as image content in `model_output` steps.

## Contents
- [Call and parse every step](#call-and-parse-every-step)
- [Prompts that worked in the notebook](#prompts-that-worked-in-the-notebook)
- [Tips](#tips)

## Call and parse every step

```python
import base64
response = client.interactions.create(
    model="gemini-robotics-er-2-preview",
    input=[{"type": "user_input", "content": [
        {"type": "image", "data": img_b64, "mime_type": "image/png"},
        {"type": "text", "text": prompt}]}],
    generation_config={"thinking_level": "high"},
    tools=[{"type": "code_execution"}],
    # optional, from the docs: system_instruction="Be precise. When JSON is requested, reply with ONLY that JSON (no preface, no code block).",
)

texts, images = [], []
for step in response.steps:
    if step.type == "code_execution_call":
        print("--- CODE ---\n", step.arguments.code)
    elif step.type == "code_execution_result":
        print("--- RESULT ---\n", step.result)
    elif step.type == "model_output":
        for c in step.content:
            if c.type == "text" and c.text:
                texts.append(c.text)
            elif c.type == "image" and c.data:
                images.append(base64.b64decode(c.data))     # e.g. annotated result
```

`response.output_text` only holds the trailing text. Iterate over `steps` to collect everything.

## Prompts that worked in the notebook

**Instrument reading with explicit procedure and answer line** (result: `Answer: 35`):

```text
How full is the meter of liquid?
To read it,
1) Find the points for the top of the sight window, bottom of the sight window and the liquid level;
2) Use math to determine the liquid level as a percentage;
3) Output "Answer: ??" on a separate line, where ?? is a number without % or unit.
```

Parse it with `re.search(r"Answer:\s*([-\d.]+)", text)`.

**Fine print** (result: `M12L64164A`):

```text
What is the number on the ESMT chip? Zoom, crop, and rotate using Python to figure it out.
```

**Annotated image** (arrows from items to the correct disposal bins):

```text
Look at this image and return it as an annotated version using arrows of different colors to represent
which items should go in which bins for disposal. You must return the final image to the API caller.
For pointing, use the format [y, x] normalized to 0-1000.
```

**Detection with self-verification** (ER crops, draws, checks and refines before answering):

```text
Return the bounding boxes in the format [ymin, xmin, ymax, xmax] normalized to 0-1000 for the compostable
objects in this scene. Please Zoom and crop the image for a clearer view. Return an annotated image of the
final result with the bounding boxes drawn on it to the API caller as a part of your process.
```

**Structured clock reading without code execution** (structured output plus high thinking):

```python
class ClockResponse(BaseModel):
    valid: bool; hours: int; minutes: int
    date: str | bool; month: str | bool; weekday: str | bool
prompt = ("What time is it? Is it a valid time that is possible? Always include validity, hours and minutes in "
          "your response. Include date, month and day of the week in your answer if they are explicitly shown.")
```

**From the docs** (code execution plus a JSON reply):

```text
Read the current value from this gauge. Then, calculate how long it will take at the current rate for the
value to reach maximum. Reply in JSON: {"current_value": val, "max_value": val, "time_to_max_minutes": val}

Measure the amount of fluid in the container. Reply in JSON:
{"fluid_level_ml": val, "container_capacity_ml": val, "percentage_full": val}

Read all visible component labels and markings on this circuit board.
Reply in JSON: {"components": [{"label": val, "location": [y, x]}]}
```

## Tips

- Give the procedure as numbered steps and require a machine-parsable final line or JSON.
- Use `thinking_level: "high"`. Code execution adds latency (several seconds or more), so use it for inspection
  steps, not inside a fast control loop.
- `response_format` (JSON schema) can be combined with `code_execution` on ER 2 when you need strict output.
- The model may emit a JSON block inside a fenced block in `model_output`. Run it through `parse_er_json`.
- In a robot pipeline, use agentic vision as a **precise re-detection** step. Get a coarse box quickly
  without tools, then ask for refinement with code execution only when confidence matters.
