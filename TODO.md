# User Study Runbook

Run all commands from the repository root:

```bash
cd /home/skim3674/Desktop/CoAssembly
```

Replace `P01` with the participant ID for the current session. Reuse the same
ID across every condition completed by that participant.

## Study 1 — Referring expressions

```bash
python3 study1_referring_expression.py
```

Open <http://127.0.0.1:5000>. Responses are saved to
`study_logs/study1/referring_expression_user.csv`.

## Robot controller — Studies 2 and 3

Start the controller in a separate terminal before launching either study.

### Real robot

```bash
python3 robot_control_server.py --no-simulation
```

### Simulation

```bash
python3 robot_control_server.py --simulation
```

## Study 2 — Workholding

Use the Unity `WorkholdingTesting` scene. Confirm that
`WorkholdingBoxReceiver` is listening on port `5026`. The target file contains
the ten board poses shared by all three conditions.

### Teach target poses

```bash
python3 workholding_study.py \
  --session-name setup \
  --mode freedrive \
  --teach-targets study_logs/study2/workholding_targets.json
```

Teaching overwrites the shared target-pose file. Skip this step during normal
participant sessions unless the targets need to be recalibrated.

### Condition 1 — Freedrive

```bash
python3 workholding_study.py \
  --session-name P01 \
  --mode freedrive \
  --target-poses-file study_logs/study2/workholding_targets.json
```

python3 workholding_study.py \
  --session-name P01 \
  --mode freedrive \
  --target-poses-file study_logs/study2/workholding_targets.json \
  --no-resume

### Condition 2 — AR handle

```bash
python3 workholding_study.py \
  --session-name P01 \
  --mode ar \
  --target-poses-file study_logs/study2/workholding_targets.json
```

### Condition 3 — Hybrid

```bash
python3 workholding_study.py \
  --session-name P01 \
  --mode hybrid \
  --target-poses-file study_logs/study2/workholding_targets.json
```

### Condition 4 — TouchGrab

Use the Unity `WorkHoldingTestNew` scene for this condition. The participant
directly grabs the cyan AR board with ISDK Touch Hand Grab; the Python data
path is otherwise identical to the AR condition. In AR, Hybrid-AR, and
TouchGrab, the virtual board moves while held and the physical robot moves
only after release.

```bash
python3 workholding_study.py \
  --session-name P01 \
  --mode touchgrab \
  --target-poses-file study_logs/study2/workholding_targets.json
```

Study 2 writes one shared replay log per participant to
`study_logs/study2/P01_replay.jsonl`; each record includes its condition in the
`mode` field. Restarting a command automatically resumes after the last fully
completed trial. Records from an unfinished trial are removed, and that trial
is repeated. Add `--no-resume` to intentionally restart only the selected
condition from Trial 1; records from the other conditions are preserved.

## Study 3 — Robot handover

Run each condition with the same participant ID. Use the real-robot controller
command above before starting.

### Condition 1 — No ghost, no color

```bash
python3 study3_handover_study.py \
  --no-simulation \
  --participant-id P01 \
  --condition no_ghost_no_color
```

### Condition 2 — Ghost, no color

```bash
python3 study3_handover_study.py \
  --no-simulation \
  --participant-id P01 \
  --condition ghost_no_color
```

<!-- ### Condition 3 — No ghost, robot color

```bash
python3 study3_handover_study.py \
  --no-simulation \
  --participant-id P01 \
  --condition no_ghost_robot_color
``` -->

### Condition 4 — Ghost and color

```bash
python3 study3_handover_study.py \
  --no-simulation \
  --participant-id P01 \
  --condition ghost_color
```

## Study 4 — Part acquisition

Run each condition with the same participant ID.

### Condition 1 — Gesture

```bash
python3 task_graph/study4_part_acquisition_study.py \
  --participant-id mahya \
  --condition gesture \
  --no-voice
```

### Condition 2 — Language and gesture

```bash
python3 task_graph/study4_part_acquisition_study.py \
  --participant-id P0 \
  --condition language \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct
```

### Condition 3 — Task-aware language and gesture

By default, an existing participant session is resumed. Use `--no-resume` only
when intentionally starting that participant and condition from scratch.

#### Resume participant session

```bash
python3 task_graph/study4_part_acquisition_study.py \
  --participant-id mahya \
  --condition task_aware \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct
```

#### Start fresh participant session

```bash
python3 task_graph/study4_part_acquisition_study.py \
  --participant-id P01 \
  --condition task_aware \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct \
  --no-resume
```




ID	Internal name	Friendly name
14	GEAR_ROD_ROW1	Row 1 gear rod
15	GEAR_ROD_ROW2	Row 2 gear rod
16	GEAR_ROD_ROW3	Row 3 gear rod
17	GEAR_ROD_ROW4	Row 4 gear rod
18	GEAR_STAND_ROW1_LEFT	Row 1 left gear stand
19	GEAR_STAND_ROW1_RIGHT	Row 1 right gear stand
20	GEAR_STAND_ROW2_LEFT	Row 2 left gear stand
21	GEAR_STAND_ROW2_RIGHT	Row 2 right gear stand
22	GEAR_STAND_ROW3_LEFT	Row 3 left gear stand
23	GEAR_STAND_ROW3_RIGHT	Row 3 right gear stand
24	GEAR_STAND_ROW4_LEFT	Row 4 left gear stand
25	GEAR_STAND_ROW4_RIGHT	Row 4 right gear stand
26	GEAR_ROW1_LEFT	Row 1 gear
27	GEAR_ROW2_LEFT	Row 2 left gear
28	GEAR_ROW2_RIGHT	Row 2 right gear
29	GEAR_ROW3_LEFT	Row 3 left gear
30	GEAR_ROW3_RIGHT	Row 3 right gear
31	GEAR_ROW4_LEFT	Row 4 gear
32	ROW3_KIT	Row 3 component kit
33	ROW4_KIT	Row 4 component kit
34	ROW2_KIT	Row 2 component kit
35	ROW1_KIT	Row 1 component kit
