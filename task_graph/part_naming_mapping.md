# Gearbox Part Naming Mapping

The gearbox uses two naming layers:

1. **Task-graph names** identify the logical parts used by
   `gearbox_task_graph.py`. `task_description.md` is written entirely in
   this vocabulary.
2. **Babylon/CSV `target_id` values** identify the corresponding STL
   rendering assets — these stay tied to the physical `.stl` files and are
   unaffected by naming-convention changes.

**CSV `target_name` values now reuse the task-graph convention directly**
(see `TASKGRAPH_NAMES` in `referring_expression_test_babylon.py` and the
evaluator's `parse_prediction()`), so no canonical-name-to-dataset-label
conversion step is needed at evaluation time.

The task graph has 43 individual-part identifiers, while the current
CSV/Babylon dataset has 26 rendering classes. Physically identical bearings
and wooden pins are grouped into shared rendering classes, and the left and
right screws from each row are grouped into one row-specific screw class.
For those grouped classes, `target_name` uses the generic root that
`task_description.md` already uses for the group (`BEARING_*` -> `BEARING`,
`PIN_*` -> `PIN`, `SCREW_ROW{n}_*` -> `SCREW_ROW{n}`) rather than one
specific row/side identifier, since the rendering asset doesn't distinguish
which position was shown.

## Individual-Part Mapping

| Task-graph name(s) | CSV `target_name` | Babylon/CSV `target_id` |
| --- | --- | --- |
| `BASE_BOARD` | `BASE_BOARD` | `BaseBoard.stl` |
| All eight `BEARING_ROW{1–4}_{LEFT/RIGHT}` parts | `BEARING` | `Bearing.stl` |
| `CRANK_HANDLE_ROW1` | `CRANK_HANDLE_ROW1` | `Handle.stl` |
| `GEAR_ROD_ROW1` | `GEAR_ROD_ROW1` | `Row1_GearRod.stl` |
| `STAND_ROW1_LEFT` | `STAND_ROW1_LEFT` | `Row1_GearStand_Left.stl` |
| `STAND_ROW1_RIGHT` | `STAND_ROW1_RIGHT` | `Row1_GearStand_Right.stl` |
| `GEAR_ROW1_LEFT` | `GEAR_ROW1_LEFT` | `Row1_Gear_Left.stl` |
| `SCREW_ROW1_LEFT`, `SCREW_ROW1_RIGHT` | `SCREW_ROW1` | `Row1_Screws.stl` |
| `GEAR_ROD_ROW2` | `GEAR_ROD_ROW2` | `Row2_GearRod.stl` |
| `STAND_ROW2_LEFT` | `STAND_ROW2_LEFT` | `Row2_GearStand_Left.stl` |
| `STAND_ROW2_RIGHT` | `STAND_ROW2_RIGHT` | `Row2_GearStand_Right.stl` |
| `GEAR_ROW2_LEFT` | `GEAR_ROW2_LEFT` | `Row2_Gear_Left.stl` |
| `GEAR_ROW2_RIGHT` | `GEAR_ROW2_RIGHT` | `Row2_Gear_Right.stl` |
| `SCREW_ROW2_LEFT`, `SCREW_ROW2_RIGHT` | `SCREW_ROW2` | `Row2_Screws.stl` |
| `GEAR_ROD_ROW3` | `GEAR_ROD_ROW3` | `Row3_GearRod.stl` |
| `STAND_ROW3_LEFT` | `STAND_ROW3_LEFT` | `Row3_GearStand_Left.stl` |
| `STAND_ROW3_RIGHT` | `STAND_ROW3_RIGHT` | `Row3_GearStand_Right.stl` |
| `GEAR_ROW3_LEFT` | `GEAR_ROW3_LEFT` | `Row3_Gear_Left.stl` |
| `GEAR_ROW3_RIGHT` | `GEAR_ROW3_RIGHT` | `Row3_Gear_Right.stl` |
| `SCREW_ROW3_LEFT`, `SCREW_ROW3_RIGHT` | `SCREW_ROW3` | `Row3_Screws.stl` |
| `GEAR_ROD_ROW4` | `GEAR_ROD_ROW4` | `Row4_GearRod.stl` |
| `STAND_ROW4_LEFT` | `STAND_ROW4_LEFT` | `Row4_GearStand_Left.stl` |
| `STAND_ROW4_RIGHT` | `STAND_ROW4_RIGHT` | `Row4_GearStand_Right.stl` |
| `GEAR_ROW4_LEFT` | `GEAR_ROW4_LEFT` | `Row4_Gear_Left.stl` |
| `SCREW_ROW4_LEFT`, `SCREW_ROW4_RIGHT` | `SCREW_ROW4` | `Row4_Screws.stl` |
| All seven `PIN_ROW*` parts | `PIN` | `WoodenPin.stl` |

## Grouping Summary

| Physical/rendering class | Task-graph identifiers represented |
| --- | --- |
| `BEARING` | Eight row-and-side-specific bearing identifiers |
| `PIN` | Seven row-and-side-specific pin identifiers |
| `SCREW_ROW1` | Row 1 left and right screw identifiers |
| `SCREW_ROW2` | Row 2 left and right screw identifiers |
| `SCREW_ROW3` | Row 3 left and right screw identifiers |
| `SCREW_ROW4` | Row 4 left and right screw identifiers |

