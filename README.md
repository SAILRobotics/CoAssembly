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
