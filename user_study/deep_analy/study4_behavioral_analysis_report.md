# Study 4 Behavioral Analysis (Objective Logs)

Participants with complete gesture/language/task_aware logs: **11** (apoorva, arunn, austin, junghoon, junhyun, kuber, mahya, mann, pantea, parisa, sandeep).
Time, interaction count, and physical effort are per-step means, normalized by each participant's completed step count so incomplete sessions aren't biased by having fewer steps to sum over.
Interaction count: Gesture counts `hover` events; Language + Gesture and Task-Aware count `vlm_interaction`/`vlm_answer` events (spoken request/response turns).

## Data completeness

Sessions with fewer than 8 completed steps (out of 8 expected):

- austin / Task-Aware: 7/8 steps
- mahya / Gesture: 7/8 steps
- mahya / Language + Gesture: 7/8 steps
- mahya / Task-Aware: 7/8 steps
- pantea / Language + Gesture: 7/8 steps
- parisa / Gesture: 7/8 steps
- parisa / Language + Gesture: 7/8 steps
- parisa / Task-Aware: 7/8 steps

## Condition summaries

| Metric | Condition | n | Mean ± SD | Median |
|---|---|---:|---:|---:|
| Time per step (s) | Gesture | 11 | 31.978 ± 15.903 | 30.373 |
| Time per step (s) | Language + Gesture | 11 | 31.281 ± 8.355 | 30.588 |
| Time per step (s) | Task-Aware | 11 | 27.812 ± 7.717 | 26.918 |
| Time per part (s) | Gesture | 11 | 13.716 ± 6.393 | 13.213 |
| Time per part (s) | Language + Gesture | 11 | 13.641 ± 5.003 | 12.205 |
| Time per part (s) | Task-Aware | 11 | 12.701 ± 4.100 | 11.939 |
| Error rate (wrong / total selections) | Gesture | 11 | 0.199 ± 0.099 | 0.188 |
| Error rate (wrong / total selections) | Language + Gesture | 11 | 0.000 ± 0.000 | 0.000 |
| Error rate (wrong / total selections) | Task-Aware | 11 | 0.000 ± 0.000 | 0.000 |
| Interactions per step | Gesture | 11 | 33.737 ± 22.463 | 31.375 |
| Interactions per step | Language + Gesture | 11 | 2.740 ± 0.489 | 2.625 |
| Interactions per step | Task-Aware | 11 | 2.729 ± 0.313 | 2.714 |
| Interactions per part | Gesture | 11 | 14.378 ± 9.419 | 13.905 |
| Interactions per part | Language + Gesture | 11 | 1.195 ± 0.349 | 1.100 |
| Interactions per part | Task-Aware | 11 | 1.239 ± 0.194 | 1.167 |
| Head translation per step (m) | Gesture | 11 | 1.390 ± 1.114 | 1.124 |
| Head translation per step (m) | Language + Gesture | 11 | 0.672 ± 0.452 | 0.622 |
| Head translation per step (m) | Task-Aware | 11 | 0.645 ± 0.592 | 0.479 |
| Hand translation per step (m) | Gesture | 9 | 2.862 ± 1.078 | 2.962 |
| Hand translation per step (m) | Language + Gesture | 9 | 1.471 ± 1.253 | 1.330 |
| Hand translation per step (m) | Task-Aware | 9 | 1.134 ± 1.330 | 0.543 |
| Head rotation per step (deg) | Gesture | 11 | 284.812 ± 258.208 | 236.403 |
| Head rotation per step (deg) | Language + Gesture | 11 | 220.621 ± 160.878 | 239.431 |
| Head rotation per step (deg) | Task-Aware | 11 | 172.537 ± 127.961 | 128.523 |

## Friedman omnibus tests

| Metric | n | χ²(2) | p | Kendall's W |
|---|---:|---:|---:|---:|
| Time per step (s) | 11 | 2.18 | 0.336 | 0.10 |
| Time per part (s) | 11 | 0.55 | 0.761 | 0.02 |
| Error rate (wrong / total selections) | 11 | 22.00 | <.001 | 1.00 |
| Interactions per step | 11 | 11.45 | 0.003 | 0.52 |
| Interactions per part | 11 | 12.18 | 0.002 | 0.55 |
| Head translation per step (m) | 11 | 7.43 | 0.024 | 0.34 |
| Hand translation per step (m) | 9 | 8.71 | 0.013 | 0.48 |
| Head rotation per step (deg) | 11 | 2.48 | 0.290 | 0.11 |

## Holm-corrected pairwise Wilcoxon tests

| Metric | Condition A | Condition B | n | A − B | W | Raw p | Holm p |
|---|---|---|---:|---:|---:|---:|---:|
| Time per step (s) | Gesture | Language + Gesture | 11 | 0.697 | 32.00 | 0.966 | 1.000 |
| Time per step (s) | Gesture | Task-Aware | 11 | 4.166 | 26.00 | 0.577 | 1.000 |
| Time per step (s) | Language + Gesture | Task-Aware | 11 | 3.469 | 17.00 | 0.175 | 0.524 |
| Time per part (s) | Gesture | Language + Gesture | 11 | 0.075 | 30.00 | 0.831 | 1.000 |
| Time per part (s) | Gesture | Task-Aware | 11 | 1.015 | 27.00 | 0.638 | 1.000 |
| Time per part (s) | Language + Gesture | Task-Aware | 11 | 0.940 | 27.00 | 0.638 | 1.000 |
| Error rate (wrong / total selections) | Gesture | Language + Gesture | 11 | 0.199 | 0.00 | <.001 | 0.003 |
| Error rate (wrong / total selections) | Gesture | Task-Aware | 11 | 0.199 | 0.00 | <.001 | 0.003 |
| Error rate (wrong / total selections) | Language + Gesture | Task-Aware | 11 | 0.000 | 0.00 | 1.000 | 1.000 |
| Interactions per step | Gesture | Language + Gesture | 11 | 30.997 | 1.00 | 0.002 | 0.006 |
| Interactions per step | Gesture | Task-Aware | 11 | 31.008 | 1.00 | 0.002 | 0.006 |
| Interactions per step | Language + Gesture | Task-Aware | 11 | 0.011 | 33.00 | 1.000 | 1.000 |
| Interactions per part | Gesture | Language + Gesture | 11 | 13.182 | 1.00 | 0.002 | 0.006 |
| Interactions per part | Gesture | Task-Aware | 11 | 13.138 | 1.00 | 0.002 | 0.006 |
| Interactions per part | Language + Gesture | Task-Aware | 11 | -0.044 | 19.00 | 0.240 | 0.240 |
| Head translation per step (m) | Gesture | Language + Gesture | 11 | 0.718 | 9.00 | 0.032 | 0.064 |
| Head translation per step (m) | Gesture | Task-Aware | 11 | 0.745 | 5.00 | 0.010 | 0.029 |
| Head translation per step (m) | Language + Gesture | Task-Aware | 11 | 0.027 | 17.00 | 0.515 | 0.515 |
| Hand translation per step (m) | Gesture | Language + Gesture | 9 | 1.392 | 7.00 | 0.074 | 0.148 |
| Hand translation per step (m) | Gesture | Task-Aware | 9 | 1.729 | 2.00 | 0.012 | 0.035 |
| Hand translation per step (m) | Language + Gesture | Task-Aware | 9 | 0.337 | 10.00 | 0.499 | 0.499 |
| Head rotation per step (deg) | Gesture | Language + Gesture | 11 | 64.191 | 24.00 | 0.465 | 0.465 |
| Head rotation per step (deg) | Gesture | Task-Aware | 11 | 112.275 | 11.00 | 0.054 | 0.161 |
| Head rotation per step (deg) | Language + Gesture | Task-Aware | 11 | 48.084 | 10.00 | 0.139 | 0.277 |
