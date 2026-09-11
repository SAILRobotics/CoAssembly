## Run the Study 2 efficiency analysis

```bash
cd /home/skim3674/Desktop/CoAssembly

python3 study2_replay_to_csv.py \
  study_logs/study2/apoorva_replay.jsonl \
  study_logs/study2/austin_replay.jsonl \
  study_logs/study2/junghoon_replay.jsonl \
  study_logs/study2/junhyun_replay.jsonl \
  study_logs/study2/kuber_replay.jsonl \
  study_logs/study2/mann_replay.jsonl \
  study_logs/study2/pantea_replay.jsonl \
  study_logs/study2/parisa_replay.jsonl \
  study_logs/study2/sandeep_replay.jsonl \
  study_logs/study2/arunn_replay.jsonl \
  --csv study_logs/study2/study2_efficiency.csv
```

## Mark rotational mistakes / orientation confusion

Run the annotator separately for each participant:

```bash
cd /home/skim3674/Desktop/CoAssembly

# python3 study2_annotate_grasp.py apoorva --. ar handle 
# python3 study2_annotate_grasp.py austin
# python3 study2_annotate_grasp.py junghoon
# python3 study2_annotate_grasp.py junhyun
# python3 study2_annotate_grasp.py kuber
# python3 study2_annotate_grasp.py mann
# python3 study2_annotate_grasp.py pantea

# python3 study2_annotate_grasp.py sandeep

# python3 study2_annotate_grasp.py parisa
```

With the 3D viewer focused:

1. Move the timeline slider to the beginning of the rotational mistake.
2. Press `C` to mark the orientation-noise start.
3. Move the slider to the end of the mistake.
4. Press `C` again to save the interval.
5. Press `V` to cancel a pending start or undo the most recent interval.
6. Press `N`/`B` to save and change trials, or `Q`/`Esc` to save and quit.
7. Press `H` at any time to print all annotation keyboard controls.

Marked intervals appear in magenta and are saved separately as:

```text
study_logs/study2/<participant>_orientation_noise_annotations.json
```

The replay logs and original grasp-annotation files are not modified.
