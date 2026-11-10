# User-study tools

Run commands from the repository root so data continues to resolve under
`user_study/study_logs/` and the shared project modules remain available.

## Execution

Programs used while conducting a study are in `execution/`, for example:

```bash
python3 -m user_study.execution.study1_referring_expression
python3 -m user_study.execution.study3_handover_study --help
python3 -m user_study.execution.workholding_study --help
python3 -m user_study.execution.workholding_testing --help
python3 -m user_study.execution.study4_part_acquisition_study --help
```

## Analysis

Replay, annotation, conversion, and evaluation programs and their example data
are in `analysis/`, for example:

```bash
python3 -m user_study.analysis.study_replay --help
python3 -m user_study.analysis.study2_replay_to_csv --help
python3 -m user_study.analysis.study2_annotate_grasp --help
python3 -m user_study.analysis.evaluate_referring_expression_models --help
```
