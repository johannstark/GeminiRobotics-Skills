# Video understanding: analysis, moment finding, progress (notebook section 5)

These features need **`gemini-robotics-er-2-preview`**. They are not available on ER 1.6 or the streaming model.
Sources are notebook section 5 and <https://ai.google.dev/gemini-api/docs/robotics-video-progress>.

Use them to:

- **verify success** without hand-written heuristics
- **find the moment** an event happened (grasp, release, insertion) to sequence the next step or trim datasets
- **grade progress** to decide whether to continue, retry, or escalate

## Contents
- [Get the video in](#get-the-video-in)
- [A. Timestamped analysis plus follow-ups](#a-timestamped-analysis-plus-follow-ups)
- [B. Moment finding](#b-moment-finding)
- [C. Progress classification](#c-progress-classification)
- [Closing the loop in a simulator](#closing-the-loop-in-a-simulator)
- [Tips](#tips)

## Get the video in

**Record:**

- Python MuJoCo: `mujoco.Renderer` plus `imageio` (`mujoco-physics/scripts/panda_pick_place.py --video`).
- Browser: `canvas.captureStream()` with `MediaRecorder` (see the `threejs-rendering` skill).
- Real robot: record the camera streams.

**Multi-camera grids** (several views tiled into each frame) work well. The notebook videos use 4–5 views.
**Describe the grid order in the prompt.** Use 10 FPS or more when recording, but note the API samples at **1 FPS by default**.

**Upload once and poll until ready:**

```python
import time
video = client.files.upload(file="episode.mp4")
while video.state.name == "PROCESSING":
    time.sleep(2)
    video = client.files.get(name=video.name)
if video.state.name == "FAILED":
    raise ValueError(f"File processing failed: {video.uri}")
VIDEO = {"type": "video", "uri": video.uri, "mime_type": video.mime_type}
# Fast motion? Sample more frames:  VIDEO["processing"] = {"type": "static", "fps": 4}
```

```ts
let video = await client.files.upload({ file: blobOrPath, config: { mimeType: 'video/webm' } });  // or 'video/mp4'
while (video.state === 'PROCESSING') { await new Promise(r => setTimeout(r, 2000)); video = await client.files.get({ name: video.name! }); }
const VIDEO = { type: 'video', uri: video.uri, mime_type: video.mimeType };
```

`er_tools.upload_and_wait()` and `er_tools.video_part()` wrap this.

## A. Timestamped analysis plus follow-ups

The server keeps state, so you can ask follow-ups **without re-uploading**:

```python
from typing import List
class VideoEvent(BaseModel):
    start_timestamp: str
    end_timestamp: str
    description: str
class VideoAnalysis(BaseModel):
    events: List[VideoEvent]
FMT = {"type": "text", "mime_type": "application/json", "schema": VideoAnalysis.model_json_schema()}

r1 = client.interactions.create(model=MODEL_ID, response_format=FMT, input=[{"type": "user_input", "content": [
    VIDEO, {"type": "text", "text": "Describe in detail each step of finishing the task. Break it down by timestamp."}]}])

r2 = client.interactions.create(model=MODEL_ID, response_format=FMT, previous_interaction_id=r1.id,
    input=[{"type": "user_input", "content": [{"type": "text",
        "text": "Zoom into second 15 to 22 and provide a per-second breakdown of what is happening."}]}])
```

Notebook output (desk-organization video, three arms):

```json
{"events": [
  {"start_timestamp": "00:00", "end_timestamp": "00:05", "description": "The left robotic arm picks up the green marker from the table and places it into the brown tray."},
  {"start_timestamp": "00:13", "end_timestamp": "00:21", "description": "The left robotic arm picks up the blue pen from the table and places it into the black pen holder."},
  {"start_timestamp": "00:21", "end_timestamp": "00:27", "description": "The middle robotic arm picks up the red pen from the table and places it into the black pen holder."}]}
```

The follow-up returned a per-second breakdown from 00:15 to 00:22: lower, grip, lift, align, insert, release, retract.

## B. Moment finding

Notebook version: a structured `MM:SS` answer, or `-1`.

```python
class MomentResponse(BaseModel):
    final_answer: str = Field(description="The timestamp, e.g. 00:06, or -1")

MOMENT_FINDING_PROMPT = """
You will observe a robot or a human attempting to complete a task. Each frame is a grid of camera views, arranged left-to-right, top-to-bottom in the following order:
cage_back_rgb_img, cage_left_rgb_img, cage_right_rgb_img, wrist_back_left_rgb_img, wrist_back_right_rgb_img.
The task is: "drop the held screw driver on the right side of the screw driver on the table".
Please identify the exact timestamp where the robot or human successfully completes the task.
"""

r = client.interactions.create(
    model=MODEL_ID,
    input=[{"type": "user_input", "content": [VIDEO, {"type": "text", "text": MOMENT_FINDING_PROMPT}]}],
    generation_config={"thinking_level": "high"},
    response_format={"type": "text", "mime_type": "application/json", "schema": MomentResponse.model_json_schema()},
)
ts = json.loads(r.output_text)["final_answer"]        # notebook: "00:09"; "-1" means not completed
seconds = mmss_to_seconds(ts)                          # er_tools: None for "-1"
```

Docs version: seconds as a float, or `null` when the task is not completed.

```text
At what timestamp (in seconds) does the task reach successful completion?
Return a JSON object: {"completion_time_seconds": <float>}.
If the task is not completed, return {"completion_time_seconds": null}.
```

To grab the frame:

```python
cap = cv2.VideoCapture(path)
cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
ok, frame = cap.read()
```

**Adapt it for your sim:** replace the grid description with your layout. For example, with the
script's recorder: "Each frame shows 2 camera views side by side: left = top camera, right = free camera."

## C. Progress classification

Two scales. Both prompts are verbatim from the notebook; substitute `TASK` and the camera description.

```python
class ProgressResponse(BaseModel):
    digit_choice: int = Field(description="The integer choice")
    reasoning: str
FMT = {"type": "text", "mime_type": "application/json", "schema": ProgressResponse.model_json_schema()}

TASK = "insert the toy screw into the top left hole on the workbench"

PERCENTAGE_PROGRESS_PROMPT = (
    "In the following task, you will see a robot or human trying to perform a task to "
    f"{TASK}.\n\n"
    "You will see multiple camera views of the same scene. Some cameras are static and are "
    "mounted outside of the scene and some cameras are mounted on the robot arms and thus "
    "they are moving during the episode.\n\n"
    "Each image shows 4 camera views concatenated together.\n\n"
    "Considering the progress shown in the video wherein the last image in the video sequence "
    "shows the current state of the task, what is the progress percentage at the end of the "
    f"episode towards the task \"{TASK}\"?\n\n"
    "Answer with choices:\n1) 0-20\n2) 21-40\n3) 41-60\n4) 61-80\n5) 81-100")

SEMANTIC_PROGRESS_PROMPT = (
    "You are an expert video rater. In this task, you will be shown a video of a robot or human performing a task. "
    f'The robot or human was asked to do the following task: "{TASK}".\n'
    "How much progress did the robot or human make in completing the given task at the end of the video?\n"
    "Use the provided choices to answer the question. Possible answer choices:\n"
    "(1) No success: The final state shows no goal-relevant change for the task command.\n"
    "(2) Partial completion: The final state shows good progress but violates major requirements or multiple requirements.\n"
    "(3) Near completion: The final state is correct in region and intent but misses a single minor requirement.\n"
    "(4) Perfect completion: The final state satisfies all the requirements.")

def classify(prompt):
    r = client.interactions.create(model=MODEL_ID, generation_config={"thinking_level": "high"}, response_format=FMT,
        input=[{"type": "user_input", "content": [VIDEO, {"type": "text", "text": prompt}]}])
    return ProgressResponse.model_validate_json(r.output_text)
```

Notebook results on the same video:

| Scale | Answer | Reasoning (abridged) |
|---|---|---|
| Percentage (1–5) | **5** (81–100%) | "screw is already aligned with and inserted into the target hole" |
| Semantic (1–4) | **3** (near completion) | "it does not release the gripper by the end of the video" |

**The semantic rubric is stricter.** Use `digit_choice == 4` as the success gate. Use the
percentage scale for progress bars, early stopping, and "is it still making progress?" checks.

The docs' minimal variant uses brackets instead of digits:

```text
Watch this video and classify the task progress level at the final frame.
Return a JSON object with the progress bracket:
{"progress_level": "0-20" | "20-40" | "40-60" | "60-80" | "80-100"}.
```

Add `"enum": ["0-20", "20-40", "40-60", "60-80", "80-100"]` to the schema to force a valid value.

## Closing the loop in a simulator

```python
for attempt in range(3):
    ok, msg = pick_and_place(model, data, pick, place, record)        # record frames
    iio.imwrite("episode.mp4", np.stack(frames), fps=10, codec="libx264"); frames.clear()
    v = upload_and_wait(client, "episode.mp4")
    sem = classify_with(v, SEMANTIC_PROGRESS_PROMPT)                   # digit_choice 1..4
    if sem.digit_choice == 4:
        break
    log(attempt, msg, sem.reasoning)                                   # reasoning explains what's missing
    pick = redetect(...)                                               # fresh image, fresh ER points
```

Also cross-check with ground truth where you have it (sim contacts and poses). Disagreements between
ER and physics show where your prompts or camera views need work.

## Tips

- Use `thinking_level: "high"` for moment finding and progress (the notebook does this for both).
- Keep clips short and focused on one task. One video per request.
- Make the end state visible: hold the final pose for 1–2 s before you stop recording, and don't end the clip
  mid-motion. The model grades "the last image in the video sequence".
- Note that 1 FPS sampling can miss fast events. Raise `processing.fps`, or slow the playback.
- Timestamps: use `MM:SS` at ≤1 FPS sampling, and `MM:SS.sss` above 1 FPS.
- Stream (`stream=True`) for long videos to avoid timeouts.
- Delete uploads when done (`client.files.delete(name=...)`). Files also expire on their own.
