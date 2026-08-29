1. Change the parts to row1_kit, row2_kit, row3_kit, ... --> Done
2. Default pose for a specific step --> Done
3. Removal of parts / tools from pegboard based on not only handed over part --> Done
4. Right Side GearStand first --> Done

5. Referring expression output template and answer and then highlight as well

5. Pegboard parts/tools bounding boxes adjustment --> Dante
6. bounding boxes around tools/parts (more transparent) --> Dante
7. TCP mismatch / Robot Mismatch (Unsolved)
8. Take many pictures from different angles for each step --> VLM feeding --> Not needed

9. check the referring expressions.csv (s) --> Dante
# User-study operation

Run commands from the `CoAssembly` directory and from the project environment
that contains Open3D, PyBullet, OpenCV, and the robot dependencies.

## Study 1 — Referring expressions

Study 1 presents 26 individual gearbox parts twice, for 52 responses per
participant. Every participant receives the same seeded random part order.

Start the local Babylon/Flask study server:

```bash
python3 study1_referring_expression.py
```

Then open the following address in a browser on the same computer:

```text
http://127.0.0.1:5000
```

### Study 1 procedure

1. Enter the participant identifier.
2. Describe the highlighted part without using its assigned name.
3. Select **Save & continue** or press `Ctrl+Enter`/`Cmd+Enter`.
4. Complete both fixed-order rounds of 26 parts.
5. Before the next participant, select **Restart sampling** to clear the browser
   counters and restart the identical sequence from its beginning.

The **Descriptions per part** value is fixed at two. Clicking a different part
in the assembled model does not alter the assigned sequence. **Skip to next
part** omits the current target and should be used only for experimenter
recovery.

### Study 1 output

- `task_graph/referring_expression_responses.csv`: primary response table.
- `task_graph/referring_expression_responses.xlsx`: regenerated workbook when
  `openpyxl` is installed.
- `task_graph/referring_expression_renderings/`: participant-view captures.
- `task_graph/referring_expression_detection_renderings/`: annotated review
  captures.

The legacy `referring_expression_test_babylon.py` filename remains as a
compatibility launcher, but `study1_referring_expression.py` is canonical.

## Robot controller

Start exactly one controller in its own terminal before launching Study 2 or
Study 3:

```bash
# Real UR robot
python3 robot_control_server.py --no-simulation

# PyBullet simulation
python3 robot_control_server.py --simulation
```

Study 2 automatically connects to whichever controller is running. For Study 3,
match the study program's `--simulation`/`--no-simulation` flag to the controller.
Before a real-robot session, verify the calibrated base pose, workspace boundary,
clear workspace, gripper connection, and emergency-stop access.

## Study 2 — Workholding

Study 2 compares three board-placement interaction modes:

- `freedrive`: physically reposition the robot and board.
- `ar`: release an AR handle target; the robot moves to it autonomously.
- `hybrid`: both freedrive and AR control are available.

Use one stable participant/session identifier and the same seed and target file
for all three modes. Counterbalance the order of the three modes separately.

### Real robot

```bash
python3 workholding_study.py \
  --session-name P01 \
  --mode freedrive \
  --seed 0 \
  --target-poses-file task_graph/workholding_targets.json

python3 workholding_study.py \
  --session-name P01 \
  --mode ar \
  --seed 0 \
  --target-poses-file task_graph/workholding_targets.json

python3 workholding_study.py \
  --session-name P01 \
  --mode hybrid \
  --seed 0 \
  --target-poses-file task_graph/workholding_targets.json
```

### Study controls

- Show marker 100 to lock the world; `Enter` manually locks or relocks it.
- Insert/attach the workholding board when prompted.
- `S`: start, pause, or resume trial recording.
- `P` / `N` or arrow keys: inspect the previous or next target.
- `F`: force-complete a running trial for experimenter recovery.
- `Esc`: exit and flush the logs.

The default `--target-navigation preview` makes P/N change only the displayed
target. Add `--target-navigation move` only when P/N should command physical
robot motion. Do not add `--no-calibrated-robot-base` unless the experiment is
intentionally using an uncalibrated robot base.

Reusing the same session name, mode, seed, and target file resumes at the first
unfinished trial. Do not reuse a participant identifier for a different target
order.

### Study 2 output

Files are written to `study_logs/study2/`:

- `P01-<mode>_trials.csv`: completed-trial outcomes and final errors.
- `P01-<mode>_trajectory.csv`: 10 Hz robot joints and TCP trajectory.
- `P01-<mode>_replay.jsonl`: synchronized 30 Hz replay frames and interactions.

Use `--out-dir PATH` to change all Study 2 output paths, or `--replay-log FILE`
to override only the JSONL replay path.

## Study 3 — Robot handover

Study 3 uses ten trials: four right-side gear stands followed by all six gears.
Run one condition per session:

- `continuous_no_preview`: continuously track the right hand without showing a
  ghost gripper. Clicking the robot freezes the offer and arms pull-to-release;
  a second click before pulling resumes hand tracking.
- `stability_committed_preview`: commit a proposed handover target after the
  hand remains stable for 1.0 second. Clicking the ghost resets the proposal.
  An unreachable proposal is red.
- `continuous_preview`: continuously track the right hand while showing the
  live ghost-gripper target.

### Real robot

Replace `P01` with the participant identifier and run the assigned condition:

```bash
python3 study3_handover_study.py \
  --no-simulation \
  --participant-id P01 \
  --condition continuous_no_preview

python3 study3_handover_study.py \
  --no-simulation \
  --participant-id P01 \
  --condition stability_committed_preview

python3 study3_handover_study.py \
  --no-simulation \
  --participant-id P01 \
  --condition continuous_preview
```

For a simulated test, change `--no-simulation` to `--simulation` and run the
robot controller with `--simulation` too.

### Study 3 controls and flow

1. Show marker 100 until the world locks. The saved pegboard pose is loaded by
   default when marker 100 locks.
2. Press `Space` to start the next trial.
3. The robot performs the recorded approach, grasp, and retract sequence. In
   preview conditions, the ghost appears as soon as the grasp succeeds.
4. After retracting, the robot moves to the fixed handover staging joints
   `[-100.16, -84.88, -136.53, -138.25, -99.46, -185.97]` degrees.
5. The selected condition controls the handover approach.
6. Pull the object to make the gripper open and complete the transfer.
7. The robot returns to `ROBOT_DEFAULT_JOINT_DEG`; the next trial then becomes
   available.

Clicking the physical/Unity robot is an interaction command, not an application
exit. `Esc` exits the study and flushes its logs.

### Study 3 output

Files are written to `study_logs/study3/`:

- `handover_events.csv`: trial and interaction events.
- `handover_replay.jsonl`: synchronized robot, tracking, target, color, and
  interaction replay data.

Use `--study3-log FILE` and `--study3-replay-log FILE` to override these paths.

## Replay Study 2 or Study 3

`study_replay.py` detects the JSONL schema automatically:

```bash
# Study 2
python3 study_replay.py \
  study_logs/study2/P01-freedrive_replay.jsonl

# Study 2 at half speed
python3 study_replay.py \
  study_logs/study2/P01-ar_replay.jsonl \
  --speed 0.5

# Study 3
python3 study_replay.py \
  study_logs/study3/handover_replay.jsonl
```

When a JSONL file contains several appended runs, the newest session is replayed
by default. Pass `--session-id SESSION_ID` to select an older run.

## Study 4 — Standalone part acquisition

Study 4 measures how quickly and accurately participants request or select the
correct pegboard object. It does not move the robot and does not require
`main_with_robot.py` or `robot_control_server.py`.

The Study 4 process starts its Unity gearbox controller and embedded Open3D
pegboard/BoardAR mirror by default.

```bash
# C1: gesture only
python3 task_graph/study4_part_acquisition_study.py \
  --condition gesture \
  --no-voice

# C2: language grounding + gesture, without task-state validation
python3 task_graph/study4_part_acquisition_study.py \
  --condition language \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct

# C3: task-aware language + gesture
python3 task_graph/study4_part_acquisition_study.py \
  --condition task_aware \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct
```

Fetch-like language such as "get," "give," or "bring" is treated as an object
request: the object is grounded, highlighted, and logged, but no confirmation or
robot command is generated. Pegboard highlights and BoardAR component highlights
are applied directly in the embedded Open3D scene and sent to Unity.

Useful options:

- `--no-open3d-scene`: run without the experimenter Open3D mirror.
- `--no-controller`: do not start the Unity gearbox controller.
- `--no-tts`: disable spoken responses.
- `--assistant-log FILE`: override the default interaction CSV.

The default log is `study_logs/study4/assistant_interactions.csv`.
