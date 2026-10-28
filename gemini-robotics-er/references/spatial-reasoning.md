# Spatial reasoning with Gemini Robotics-ER 2 (notebook section 2)

These are point, box, trajectory and spatial-logic queries. Every example below follows a cell in
the [official ER 2 notebook](https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb) or a page at <https://ai.google.dev/gemini-api/docs/robotics-spatial>.

## Contents
- [Shared setup](#shared-setup)
- [Prompt catalogue](#prompt-catalogue)
- [Parallel part queries](#parallel-part-queries)
- [Multi-view success detection](#multi-view-success-detection)
- [Tracking in video](#tracking-in-video)
- [Precision: consensus and cropping](#precision-consensus-and-cropping)
- [From 2D to a grasp](#from-2d-to-a-grasp)

## Shared setup

```python
import base64, json
from io import BytesIO
from typing import List
from PIL import Image
from pydantic import BaseModel, Field
from google import genai

client = genai.Client()
MODEL_ID = "gemini-robotics-er-2-preview"

def load_image(path):                       # notebook helper: 800 px wide PNG
    img = Image.open(path)
    img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)
    buf = BytesIO(); img.save(buf, format="PNG")
    return img, base64.b64encode(buf.getvalue()).decode()

class PointItem(BaseModel):
    point: List[int] = Field(description="[y, x] format normalized to 0-1000")
    label: str = Field(description="Identifying name for the object detected")
class PointResponse(BaseModel):
    items: List[PointItem]

class BBoxItem(BaseModel):
    box_2d: List[int] = Field(description="[ymin, xmin, ymax, xmax] format normalized to 0-1000")
    label: str = Field(description="Identifying name for the object detected")
class BBoxResponse(BaseModel):
    items: List[BBoxItem]

def ask(img_b64, prompt, schema=PointResponse, thinking="low"):
    r = client.interactions.create(
        model=MODEL_ID,
        input=[{"type": "user_input", "content": [
            {"type": "image", "data": img_b64, "mime_type": "image/png"},
            {"type": "text", "text": prompt}]}],
        generation_config={"thinking_level": thinking},
        response_format={"type": "text", "mime_type": "application/json", "schema": schema.model_json_schema()},
    )
    return json.loads(r.output_text.strip())
```

To draw results: `px = x/1000*W` and `py = y/1000*H`. For boxes, use `(xmin, ymin)` as the corner and
`(xmax-xmin, ymax-ymin)/1000*(W, H)` as the size.

## Prompt catalogue

| Task | Prompt (verbatim from the notebook or docs) | Schema | Thinking |
|---|---|---|---|
| Point to everything | `Point to all the items in the image. The label returned should be an identifying name for the object detected.` | Points | low |
| Named objects | `Get all points matching the following objects: bread, starfruit, banana. The label returned should be an identifying name for the object detected.` | Points | high |
| Abstract category | `Point at all the fruits. The label returned should be the actual name of the object detected.` | Points | high |
| Every instance | `Get all points matching the game board slots.` | Points | high |
| Counting by pointing | `Point to each washer in the box.` → `len(items)` (the notebook got 16) | Points | high |
| Boxes | `Return bounding boxes with labels. Limit to 25 objects. Include as many objects as you can identify on the table.` | Boxes | low |
| Affordance / intent | `"I need to blow my nose." Find the cubby that can help.` | Boxes | low |
| Trajectory over an area | `Point to the blue brush and a list of 10 points covering the region of particles. Ensure that the points are spread evenly over the particles to create a smooth trajectory. Label the points from 1 to 10 based on the order that they should be approached.` | Points | low |
| Collision-free path | `Find the most direct collision-free trajectory of 10 points on the floor between the current view origin and the green ottoman in the back left. The points should avoid all other obstacles on the floor.` | Points | low |
| Make room | `Point to the object that I need to remove to make room for my laptop.` | Points | high |
| Plan + point | `Explain how to pack the lunch box and lunch bag. Point to each object that you refer to.` | `{items: [...], answer: str}` | high |
| Free slots | `Point to the unobstructed empty sockets.` | Points | high |
| Payload reasoning | `I am a robot with a payload of 3LBs. Point to ALL the items in the image I am physically able to pick up.` | Points | high |
| Pick-arm waypoints | `Generate a trajectory for the robotic arm to pick up the red pen and place it in the organizer. Return a list of waypoints as JSON: [{"step": <int>, "point": [y, x], "action": <description>}, ...] where coordinates are normalized to 0-1000.` | free JSON | medium |

Prompts used in the Franka demo (`App.tsx::defaultPromptParts`):

```text
Identify {subject} in the scene and mark them with points. DO NOT mark items that only match the description
partially. Follow the JSON format: [{"point": [y, x], "label": "label"}, ...]. The points are in [y, x] format normalized to 0-1000.

Detect {subject}, with no more than 25 items. DO NOT detect items that only match the description partially.
Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".
```

More robot-specific templates:

```text
Point to the best place to grasp the {object} with a parallel-jaw gripper from above. Only return one point.
Point to the empty area inside the {container} where a {object} can be placed. Only return one point.
Point to the {part} of the {object} in the image. Only return one point.     # e.g. handle of the mug
```

> The docs' own examples sometimes request other box layouts, such as `{"y","x","y2","x2"}`. Pick one
> layout, put it in **both** the prompt and the schema, and parse only that layout.

## Parallel part queries

Independent queries can run concurrently. The notebook does this to cut latency:

```python
import concurrent.futures
queries = [("paper bag", "handles"), ("banana", "the stem"), ("measuring cup", "handle"), ("bowl", "tomato")]

def part(obj, p):
    try:
        return ask(img_b64, f"Point to the {p} of the {obj} in the image. Only return one point.").get("items", [])
    except Exception:
        return []

with concurrent.futures.ThreadPoolExecutor() as ex:
    points = [pt for res in ex.map(lambda q: part(*q), queries) for pt in res]
```

## Multi-view success detection

ER compares several images in a single request (notebook section 2, "Multi-View Correspondence"):

```python
class SuccessResponse(BaseModel):
    success: bool = Field(description="True if the robot successfully performed the task, false otherwise")

prompt = (
    "For this task, you will see a robot or human trying to perform the task of putting the mango into the brown container. "
    "You may see multiple camera views of the same scene.\n\n"
    "The first 4 images show multiple camera views from the start of the episode (some time ago). "
    "The last 4 images show multiple camera views from the current moment in the episode (as it is now).\n\n"
    "Looking at these images and comparing the start of the episode with current state, did the robot successfully perform the task?")

content = [{"type": "image", "data": load_image(p)[1], "mime_type": "image/png"}
           for p in [*initial_views, *current_views]] + [{"type": "text", "text": prompt}]
r = client.interactions.create(model=MODEL_ID, input=[{"type": "user_input", "content": content}],
                               generation_config={"thinking_level": "high"},
                               response_format={"type": "text", "mime_type": "application/json",
                                                "schema": SuccessResponse.model_json_schema()})
success = json.loads(r.output_text)["success"]          # the notebook run returned False
```

This is a cheap post-action check that needs no video: capture the same cameras before and after each
pick. **State the image order explicitly.**

## Tracking in video

```python
f = client.files.upload(file="my-video.mp4")      # then poll until ACTIVE (see video-progress.md)
r = client.interactions.create(model=MODEL_ID, input=[
    {"type": "video", "uri": f.uri, "mime_type": f.mime_type},
    {"type": "text", "text": 'Point to the red ball in every frame where it appears. The answer should follow the json '
                             'format: [{"point": [y, x], "label": <label>}, ...]. The points are in [y, x] format '
                             'normalized to 0-1000. Return one entry per frame that contains the object.'}])
```

## Precision: consensus and cropping

- **Median of N.** Run the same query 3–5 times, in parallel, and take the per-axis median. Discard outliers more
  than about 30 units from the median.
- **Crop and zoom.** For small targets, crop around a coarse detection, re-query on the crop, and map back:
  `y_full = y0 + y_crop/1000*h_crop`, then renormalize by the full image size. Or let ER do this itself
  with code execution (see `agentic-vision.md`).
- **Media resolution.** `{"type": "image", ..., "resolution": "high"}` (`low`/`medium`/`high`/`ultra_high`)
  trades tokens for detail.
- Good lighting and colour contrast matter, both in the sim (materials and lights) and on real cameras.

## From 2D to a grasp

1. Point or box centre `[y, x]` → pixel `(u, v)` in the **image that was sent**. If you resized or
   cropped before sending, map back first.
2. Pixel → 3D: raycast (browser sim), depth deprojection (Python sim or RGB-D), or a table-plane intersection with an ArUco mat (see also `real-camera-grounding.md`).
3. Top-surface hit → grasp height: subtract about half the object height. The Franka demo uses −2 cm for 4 cm cubes.
4. Yaw from the box: if the box is wider than it is tall in the image, close the gripper across the image y
   axis, and vice versa. For a better result, ask ER for two points on opposite grasp edges.
5. Reject targets outside the workspace or on the robot itself. In Three.js, check `userData.bodyID`;
   in MuJoCo, check the segmentation image.
6. **Re-detect after each pick** in clutter, because objects move. The Franka demo detects once and executes blind.
