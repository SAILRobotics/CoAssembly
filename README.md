# CoAssembly
Coassembly Setup with Meta Quest3

## Gearbox assembly task graph

Run the interactive Dear PyGui task graph from the repository root:

```bash
python3 task_graph/gearbox_task_graph.py
```

The graph uses blue for ready steps, orange for blocked steps, and green for
completed steps. Select a node and click **Mark selected step complete** to
consume its input parts and create the distinctly named assembled part shown in
the step details.

The terminal panel accepts a bare part name such as `GEAR_ROD_ROW2`, or these
commands:

```text
parts
part GEAR_ROD_ROW2
step r2_gear_rod
complete r2_gear_rod
frontier
undo
status
missing
reset
help
```

`parts` lists only active parts. Completing a step removes its input components
and replaces them with the newly named assembled part. The **Active parts and
assemblies** panel shows all active items by row; assembly entries can be
expanded to inspect their nested composition. Undoing an assembly removes that
entry and restores its inputs as separate active items. Any completed frontier
step—one whose output has not been consumed by another completed step—may be
undone by selecting its green node or by entering `undo <step>`.
If a requested step is out of order, the terminal reports the inputs that are
still missing.

Run the dependency/state-engine check without opening a window:

```bash
python3 task_graph/gearbox_task_graph.py --self-test
```

## Upload images to Roboflow

Set your private API key in the environment, then upload a folder to an
existing Roboflow project:

```bash
export ROBOFLOW_API_KEY="your-private-api-key"
python3 task_graph/upload_roboflow.py /path/to/images \
  --project your-project-id \
  --batch coassembly-captures
```

The images appear in the project's **Annotate** tab. Use `--recursive` to scan
subfolders, `--tag TAG` to add a tag (repeatable), or `--split valid`/`test` to
change the dataset split. Preview the files without uploading or requiring an
API key:

```bash
python3 task_graph/upload_roboflow.py /path/to/images \
  --project your-project-id \
  --recursive \
  --dry-run
```

Run `python3 task_graph/upload_roboflow.py --help` for every option. You can
temporarily paste the key into `ROBOFLOW_API_KEY` near the top of the script;
leave it empty to read the key from the environment instead. Do not commit a
private API key to source control.

## SAM 3-assisted image annotation

Launch the Dear PyGui bounding-box annotator on the Dante captures:

```bash
python3 task_graph/sam3_annotator.py
```

The eight classes are `gear`, `gear_rod`, `gear_stand`, `baseboard`, `left_hand`,
`right_hand`, `tool`, and `bearing`. Drag on the image to create a box, click a
box to select it, press 1–8 to change the active class, and use A/D to move
between images. A selected box has eight resize handles; drag a corner or edge
handle to resize it, or drag inside the selected box to move it.
Labels are auto-saved while navigating to `task_graph/dante_captures_labels`
in YOLO detection format.

The annotator loads every third valid image from the continuous capture
sequence by default. Use `--frame-step 1` to load all images or another positive
value to change the sampling stride.

For SAM 3 suggestions, install a current Transformers release and authenticate
with Hugging Face:

```bash
pip install -U transformers accelerate
hf auth login
```

Request and accept access to `facebook/sam3` on Hugging Face. The model is then
downloaded and cached automatically on first use; no local `sam3.pt` file is
required.
Click **Suggest with SAM 3** to run text-prompt concept segmentation. SAM 3
caches each frame's image features so all class-alias queries reuse one feature
extraction. Suggestions replace the boxes currently shown and must be reviewed
before saving.

Enable **Auto-suggest on unlabeled images** to run SAM 3 automatically after
using Previous/Next, A/D, or index navigation. Auto-suggest skips every image
that already has a saved label file, including reviewed images with zero
boxes, so navigating cannot overwrite saved work.

Text and image-exemplar prompts can be combined. Select a reviewed box, click
**Set selected box as exemplar**, leave both prompt checkboxes enabled, and
click **Suggest with SAM 3**. Ultralytics' documented exemplar interface uses a
box on the current image; the annotator therefore does not carry exemplar boxes
across frames. Text and exemplar results are merged with same-class overlap
removal.
Text aliases are configured in `CLASS_ALIASES` near the top of
`task_graph/sam3_annotator.py`; every alias prediction maps back to one
canonical YOLO class.
