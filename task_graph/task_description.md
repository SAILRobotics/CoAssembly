# Gearbox Assembly Task

## Objective

Assemble four gearbox rows onto `BASE_BOARD`. Each row consists of a gear rod,
one or more matching gears, bearings, gear stands, wooden retaining pins, and
mounting screws. Row 1 also includes a crank handle.

Each row is identified by its corresponding color and shape. A component may
only be assembled with other components assigned to the same row. Components
labeled `LEFT` and `RIGHT` must remain on their designated sides.

## Visual Row Identification

Use the following color-and-shape associations to identify the four gear rows
and their corresponding components:

1. **Gear Row 1:** white and circle
2. **Gear Row 2:** red and triangle
3. **Gear Row 3:** green and hexagon
4. **Gear Row 4:** blue and square

Each color and each shape uniquely identifies one gear row. The user may refer
to a row by its row number, its color, its shape, or both attributes together.
For example, `Row 1`, `white row`, `circle row`, and `white circle row` all
refer to Gear Row 1.

## How to Interpret Live Task State

The application injects the current task-graph state into the VLM before it
answers a question. Treat that live state as authoritative:

- `COMPLETED` means the step has already been performed.
- `READY` means every graph prerequisite is satisfied and the step may be
  performed now.
- Any step that is neither `COMPLETED` nor `READY` is `BLOCKED` and must not be
  recommended as the next action.
- `Active assemblies` are the parts or subassemblies that currently exist in
  the inventory. Raw inputs consumed by a completed step are no longer active.
- A selected-step description may include `Blocked by`, which identifies its
  currently missing inputs or context.

Use the live `COMPLETED`, `READY`, and selected-step information instead of
guessing progress from this general task description. Do not infer additional
dependencies from prose, stage numbers, visual layout, or list order.

## Component Inventory

> **Note on part naming:** All bearings are physically identical to each other;
> all pins are physically identical to each other. The row number and side label
> in a part name (e.g. `BEARING_ROW2_LEFT`) indicate only *where* that part must
> be installed, not that it is a distinct physical part type.

### Row 1 — One Large Gear

- `BEARING_ROW1_LEFT`
- `BEARING_ROW1_RIGHT`
- `CRANK_HANDLE_ROW1`
- `GEAR_ROD_ROW1`
- `GEAR_ROW1_LEFT` — large gear
- `PIN_ROW1_LEFT`
- `PIN_ROW1_RIGHT` — this is for CRANK_HANDLE_ROW1
- `SCREW_ROW1_LEFT`
- `SCREW_ROW1_RIGHT`
- `STAND_ROW1_LEFT`
- `STAND_ROW1_RIGHT`

### Row 2 — One Small Gear and One Medium Gear

- `BEARING_ROW2_LEFT`
- `BEARING_ROW2_RIGHT`
- `GEAR_ROD_ROW2`
- `GEAR_ROW2_LEFT`
- `GEAR_ROW2_RIGHT`
- `PIN_ROW2_LEFT`
- `PIN_ROW2_RIGHT`
- `SCREW_ROW2_LEFT`
- `SCREW_ROW2_RIGHT`
- `STAND_ROW2_LEFT`
- `STAND_ROW2_RIGHT`

### Row 3 — One Small Gear and One Medium Gear

- `BEARING_ROW3_LEFT`
- `BEARING_ROW3_RIGHT`
- `GEAR_ROD_ROW3`
- `GEAR_ROW3_LEFT`
- `GEAR_ROW3_RIGHT`
- `PIN_ROW3_LEFT`
- `PIN_ROW3_RIGHT`
- `SCREW_ROW3_LEFT`
- `SCREW_ROW3_RIGHT`
- `STAND_ROW3_LEFT`
- `STAND_ROW3_RIGHT`

### Row 4 — One Large Gear

- `BEARING_ROW4_LEFT`
- `BEARING_ROW4_RIGHT`
- `GEAR_ROD_ROW4`
- `GEAR_ROW4_LEFT` — large gear
- `PIN_ROW4_LEFT`
- `SCREW_ROW4_LEFT`
- `SCREW_ROW4_RIGHT`
- `STAND_ROW4_LEFT`
- `STAND_ROW4_RIGHT`

### Base Component

- `BASE_BOARD`

## Assembly Rules and Constraints

1. Install each bearing into the stand with the same row number and side label.
   For example, `BEARING_ROW2_LEFT` must be installed in `STAND_ROW2_LEFT`.

2. Completely assemble each gear rod before installing it between its two gear
   stands:

   - Install one large gear onto `GEAR_ROD_ROW1`.
   - Install one small gear and one medium gear onto `GEAR_ROD_ROW2`.
   - Install one small gear and one medium gear onto `GEAR_ROD_ROW3`.
   - Install one large gear onto `GEAR_ROD_ROW4`.

3. Secure every gear to its gear rod using the corresponding wooden pin or
   pins. All pins are physically the same part; their names (e.g. `PIN_ROW2_LEFT`)
   only indicate which gear rod and side they belong to. The one exception is
   `PIN_ROW1_RIGHT`, which is used to secure `CRANK_HANDLE_ROW1` to
   `GEAR_ROD_ROW1` at the final Row 1 step rather than fixing a gear.

4. Fasten one bearing-and-stand assembly to `BASE_BOARD` before inserting the
   assembled gear rod. Do not fasten the second stand yet, because fixing both
   stands would leave insufficient clearance to insert the rod.

5. Insert the completed gear-rod assembly through the bearing in the fastened
   stand while fitting the unfastened bearing-and-stand assembly over the other
   end of the rod.

6. After the gear rod and second stand are in position, fasten the second stand
   to `BASE_BOARD` using its corresponding mounting screw.

7. Attach `CRANK_HANDLE_ROW1` to the designated end of `GEAR_ROD_ROW1` only
   after the rod has been installed between `STAND_ROW1_LEFT` and
   `STAND_ROW1_RIGHT`.

8. After all four rows have been installed, verify that every gear rotates
   freely and meshes correctly with the gears in adjacent rows.

## Stage Indices Used by the Interface and Controller

When the user refers to a numbered stage, interpret it using this mapping:

1. **Left bearing:** insert the row's left bearing into its left stand.
2. **Gear rod:** place the row's required gear or gears onto its rod and secure
   them with their assigned retaining pin or pins.
3. **Right bearing:** insert the row's right bearing into its right stand.
4. **First stand:** fasten the left bearing-and-stand assembly to `BASE_BOARD`
   using its assigned screw.
5. **Rod and second stand:** insert the completed gear rod through the fastened
   left stand while fitting the right bearing-and-stand assembly over the other
   end.
6. **Second stand:** fasten the right stand to `BASE_BOARD` using its assigned
   screw.
7. **Crank handle:** attach `CRANK_HANDLE_ROW1` to Row 1. This stage exists only
   for Row 1.
8. **Finish gearbox:** perform final alignment, meshing, and free-rotation
   verification. This is one global step, not a per-row step.

Therefore, Rows 2–4 use stages 1–6, Row 1 uses stages 1–7, and stage 8 applies
to the completed gearbox. A stage number identifies an operation; it does not
by itself prove that the operation is currently allowed. Check whether the
corresponding task step is `READY`.

Equivalent references should be understood together. For example, `Row 2
Stage 3`, `red Stage 3`, and `triangle right-bearing stage` all refer to
installing the right bearing into the right stand for Gear Row 2.

## Completion Criteria

The gearbox assembly is complete when:

- All eight bearings are installed in their corresponding stands.
- All gears are secured to the correct gear rods with wooden pins.
- All four assembled gear rods are supported by their corresponding stand
  pairs.
- All eight stands are secured to `BASE_BOARD`.
- `CRANK_HANDLE_ROW1` is attached to `GEAR_ROD_ROW1`.
- All gears rotate freely and mesh correctly with the adjacent gears.

## User Recommendation Preference

When recommending the next step, the user prefers to complete the gearbox row
by row:

1. Finish **all steps in Row 1** before starting any work on Row 2.
2. Finish **all steps in Row 2** before starting any work on Row 3.
3. Finish **all steps in Row 3** before starting any work on Row 4.
4. Perform the final verification step only after all four rows are fully
   assembled.

Within each row, the recommended sequence is:

1. Insert the left bearing into the left stand.
2. Insert the right bearing into the right stand (may be done in parallel
   with step 1).
3. Assemble the gear rod (may be done in parallel with steps 1–2).
4. Fasten the left (first) stand to `BASE_BOARD`.
5. Insert the assembled gear rod and fit the right (second) stand.
6. Fasten the right (second) stand to `BASE_BOARD`.
7. Row 1 only: attach `CRANK_HANDLE_ROW1` after both stands are fastened.

Apply this preference only when choosing among steps that the task graph marks
`READY`: prefer Row 1, then Row 2, then Row 3, then Row 4; within that row,
prefer the lowest stage number. This is a recommendation preference, not an
additional dependency. Any other `READY` step remains valid and may still be
performed if the user chooses it.

The preferred row-by-row order avoids interference between in-progress
assemblies on different rows and keeps the workspace organised.

## Guidance for VLM Answers

- Recommend only steps listed as `READY` in the injected live state.
- When several steps are `READY`, apply the user recommendation preference
  above, while making clear that the other `READY` steps remain valid.
- When asked why a step cannot be performed, report its live `Blocked by`
  information. Do not invent a missing prerequisite.
- When asked what happened to a raw part, explain that a completed step may
  have consumed it and transformed it into the listed active assembly.
- Use the canonical part and step names in backticks when practical, while also
  accepting ordinary phrases such as “left bearing,” “white row,” or “Stage 4.”
- Keep instructions concise and action-oriented. State the row, stage, parts,
  and expected output when those details help the user.
- If the live state and this general description appear inconsistent, report
  the inconsistency rather than silently choosing one.
