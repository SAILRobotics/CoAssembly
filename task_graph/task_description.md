# Gearbox Assembly Task

## Objective

Assemble four gearbox rows onto `BASE_BOARD`. Each row consists of a gear rod,
one or two matching gears, bearings, gear stands, wooden retaining pins, and
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

## Spatial Convention: Left, Right, Near, and Far

`LEFT` and `RIGHT` are fixed gearbox-local names. They do **not** change when
the user walks around the gearbox, rotates the AR model, or views it from the
opposite side.

Use this canonical viewpoint whenever interpreting a spatial expression:

- Look down at `BASE_BOARD` from above.
- Stand at the **Row 4 edge** of the board and face across it toward Row 1.
- The stands on the viewer's left are the `LEFT` stands and components.
- The stands on the viewer's right are the `RIGHT` stands and components.
- Row 4 is nearest the canonical viewpoint; Row 1 is farthest away.
- A name such as `STAND_ROW3_RIGHT` therefore means the Row 3 stand in the
  diagram's right column, even if AR rotation temporarily puts it on the
  user's screen-left.

```text
                              FAR SIDE / ROW 1 EDGE

                LEFT (+Y)                            RIGHT (-Y)
                   <-------------------------------------->
       +----------------------------------------------------------------------------------+
       |                                   BASE_BOARD                                     |
       |                                                                                  |
ROW 1  | [L STAND]====[    LARGE LEFT GEAR    ]====                         ====[R STAND] |----[CRANK]
WHITE  |                                                                                  |
       |                                                                                  |
ROW 2  | [L STAND]====[    SMALL LEFT GEAR    ]====[  MEDIUM RIGHT GEAR  ]  ====[R STAND] |
RED    |                                                                                  |
       |                                                                                  |
ROW 3  | [L STAND]====[    SMALL LEFT GEAR    ]====[  MEDIUM RIGHT GEAR  ]  ====[R STAND] |
GREEN  |                                                                                  |
       |                                                                                  |
ROW 4  | [L STAND]====[    LARGE LEFT GEAR    ]====                         ====[R STAND] |
BLUE   |                                                                                  |
       +----------------------------------------------------------------------------------+

                            OPERATOR / NEAR SIDE
                               ROW 4 EDGE
                       (looking forward toward Row 1)

Legend:
  L / R       = gearbox-local LEFT / RIGHT (not current screen-left/right)
  [short box] = gear stand
  =====       = the continuous gear rod running between the two stands
  [CRANK]     = the Row 1 crank handle

Bearings, pins, and screws are intentionally omitted from this spatial
overview. They remain part of the assembly procedure described below.
```

Read each row horizontally from its left stand, through its gear rod and
gear(s), to its right stand. `RIGHT` is assembled first; rotating the AR model
does not rename either side. Row 1 continues beyond its right stand to
`CRANK_HANDLE_ROW1`; `PIN_ROW1_RIGHT` secures that handle.
Rows 1 and 4 each contain one large left gear. Rows 2 and 3 each contain two
gears: a small left gear and a medium right gear. The schematic separates
labels for readability and does not represent exact spacing or gear diameter.
The size words are relative visual categories, not claims that gears sharing a
category have identical dimensions. In particular, the Row 2 and Row 3 right
gears are both called `medium` because each is larger than its row's small left
gear; the two medium gears may have different diameters from one another.

For each row, the right bearing-and-stand assembly is built first. The left
bearing-and-stand step becomes available only after the right assembly step is
complete. The diagram is schematic: it communicates identity and relative
placement, not manufacturing dimensions.

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

## Technical Part Terminology

The task graph uses short project-specific names. The following mechanical
terms may also be used when describing or referring to the parts:

- `BASE_BOARD`: base plate, mounting plate, or gearbox mounting plate.
- `BEARING_*`: radial bearing or shaft-support bearing. Visually, it is the
  small black circular ring installed in a stand.
- `STAND_*`: bearing support bracket or bearing pedestal. `Gear stand` is
  understandable in this project, but bearing support bracket is the more
  descriptive technical term because the stand locates the bearing that
  supports the gear shaft. Do not replace the canonical `STAND_*` identifiers.
- `GEAR_ROD_*`: gear shaft or transmission shaft. `Gear rod` remains the
  canonical task-graph term.
- `GEAR_*`: spur gear, meaning a cylindrical gear with straight teeth parallel
  to the shaft axis.
- `PIN_*`: wooden dowel pin or retaining dowel. It passes through a shaft hole
  to retain a gear, except `PIN_ROW1_RIGHT`, which retains the crank handle.
- `CRANK_HANDLE_ROW1`: hand crank or crank handle.
- `SCREW_*`: machine screw or mounting screw used to secure a bearing support
  bracket to `BASE_BOARD`.

`STAND_ROW1_RIGHT` is geometrically distinct from the other gear stands: its
bearing recess overlaps a separate hole that passes through the stand. The
other stands have the bearing recess without this additional penetrating hole.

### Row-Specific Screw Drives and Tools

The drive designation describes the recess in the screw head and the matching
driver. `H5`, `T25`, and `H3` are driver sizes rather than complete screw names.

| Row | Canonical target | Visible color | Technical drive description | Required driver |
| --- | --- | --- | --- | --- |
| 1 | `Row1_Screws.stl` / `SCREW_ROW1_*` | Silver | Internal-hex (hex-socket or Allen) mounting screw | 5 mm hex key (`H5`) |
| 2 | `Row2_Screws.stl` / `SCREW_ROW2_*` | Black | Internal Torx/star-drive mounting screw | Torx `T25` driver |
| 3 | `Row3_Screws.stl` / `SCREW_ROW3_*` | Black | Internal-hex (hex-socket or Allen) mounting screw | 3 mm hex key (`H3`) |
| 4 | `Row4_Screws.stl` / `SCREW_ROW4_*` | Silver | Phillips/cross-recess mounting screw | Phillips screwdriver |

Use the spelling `Phillips`, with two l's. Head profile terms such as socket
head, button head, low-profile head, or pan head describe the outer head shape;
they are separate from the internal drive type and driver size listed above.

The speech/VLM interface uses these fastening-tool labels:

| Row | Fastening stages | Semantic tool labels | Pegboard objects highlighted |
| --- | --- | --- | --- |
| 1 | 1.4 and 1.6 | `BIT_WRENCH`, `H5_HEX_BIT` | interchangeable-bit wrench and `BitHolder1` |
| 2 | 2.4 and 2.6 | `BIT_SCREWDRIVER`, `T25_TORX_BIT` | interchangeable-bit screwdriver and `BitHolder2` |
| 3 | 3.4 and 3.6 | `BIT_WRENCH`, `H3_HEX_BIT` | interchangeable-bit wrench and `BitHolder1` |
| 4 | 4.4 and 4.6 | `PHILLIPS_SCREWDRIVER` | Phillips screwdriver |

`BIT_HOLDER1` contains the H5 and H3 hex inserts. `BIT_HOLDER2` contains the
T25 Torx insert. Referring to an insert therefore highlights its correct
physical storage holder. Tools are reusable resources: completing a step does
not consume them into an assembly. A bare phrase such as "the bit holder" is
ambiguous and should prompt the user to specify Holder 1, Holder 2, or the bit.

### Row-Kit Storage

The small components are organized into one physical kit box per assembly row:

| Pegboard box | Physical contents |
| --- | --- |
| `ROW1_KIT` | two bearings, two mounting screws, two wooden pins, and the crank handle |
| `ROW2_KIT` | two bearings, two mounting screws, and two wooden pins |
| `ROW3_KIT` | two bearings, two mounting screws, and two wooden pins |
| `ROW4_KIT` | two bearings, two mounting screws, and two wooden pins |

The row and side identifiers in the task graph assign kit contents to assembly
locations; they do not imply separate pegboard boxes for every small part.
`GEAR_ROD_ROW4` has only one retaining-pin hole, so the current task graph uses
`PIN_ROW4_LEFT`; the second wooden pin physically stored in `ROW4_KIT` is a
spare and is not an input to an assembly step.

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

   `GEAR_ROD_ROW1`, `GEAR_ROD_ROW2`, and `GEAR_ROD_ROW3` each have two pin
   holes, while `GEAR_ROD_ROW4` has one pin hole.

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

1. **Right bearing:** insert the row's right bearing into its right stand.
2. **Gear rod:** place the row's required gear or gears onto its rod and secure
   them with their assigned retaining pin or pins.
3. **Left bearing:** insert the row's left bearing into its left stand. This
   step requires the row's right bearing-and-stand assembly to be complete.
4. **First stand:** fasten the right bearing-and-stand assembly to `BASE_BOARD`
   using its assigned screw.
5. **Rod and second stand:** insert the completed gear rod through the fastened
   right stand while fitting the left bearing-and-stand assembly over the other
   end.
6. **Second stand:** fasten the left stand to `BASE_BOARD` using its assigned
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
Stage 3`, `red Stage 3`, and `triangle left-bearing stage` all refer to
installing the left bearing into the left stand for Gear Row 2.

## Canonical Task-Graph Names

The names below are the exact identifiers used by `gearbox_task_graph.py`.
Treat spelling, row number, and `LEFT`/`RIGHT` suffixes as significant.

### Step IDs and User-Facing Names

#### Row 1

1. `r1_bearing_right` — Row 1.1: bearing into right stand
2. `r1_gear_rod` — Row 1.2: assemble gear rod
3. `r1_bearing_left` — Row 1.3: bearing into left stand
4. `r1_fasten_first_stand` — Row 1.4: fasten right stand
5. `r1_insert_rod_and_fit_second` — Row 1.5: insert rod and fit left stand
6. `r1_fasten_second_stand` — Row 1.6: fasten left stand
7. `r1_attach_handle` — Row 1.7: attach crank handle

#### Row 2

1. `r2_bearing_right` — Row 2.1: bearing into right stand
2. `r2_gear_rod` — Row 2.2: assemble gear rod
3. `r2_bearing_left` — Row 2.3: bearing into left stand
4. `r2_fasten_first_stand` — Row 2.4: fasten right stand
5. `r2_insert_rod_and_fit_second` — Row 2.5: insert rod and fit left stand
6. `r2_fasten_second_stand` — Row 2.6: fasten left stand

#### Row 3

1. `r3_bearing_right` — Row 3.1: bearing into right stand
2. `r3_gear_rod` — Row 3.2: assemble gear rod
3. `r3_bearing_left` — Row 3.3: bearing into left stand
4. `r3_fasten_first_stand` — Row 3.4: fasten right stand
5. `r3_insert_rod_and_fit_second` — Row 3.5: insert rod and fit left stand
6. `r3_fasten_second_stand` — Row 3.6: fasten left stand

#### Row 4

1. `r4_bearing_right` — Row 4.1: bearing into right stand
2. `r4_gear_rod` — Row 4.2: assemble gear rod
3. `r4_bearing_left` — Row 4.3: bearing into left stand
4. `r4_fasten_first_stand` — Row 4.4: fasten right stand
5. `r4_insert_rod_and_fit_second` — Row 4.5: insert rod and fit left stand
6. `r4_fasten_second_stand` — Row 4.6: fasten left stand

#### Global Finish

8. `finish_gearbox` — Verify and finish gearbox

### Raw Part Names

```text
BASE_BOARD

BEARING_ROW1_LEFT
BEARING_ROW1_RIGHT
CRANK_HANDLE_ROW1
GEAR_ROD_ROW1
GEAR_ROW1_LEFT
PIN_ROW1_LEFT
PIN_ROW1_RIGHT
SCREW_ROW1_LEFT
SCREW_ROW1_RIGHT
STAND_ROW1_LEFT
STAND_ROW1_RIGHT

BEARING_ROW2_LEFT
BEARING_ROW2_RIGHT
GEAR_ROD_ROW2
GEAR_ROW2_LEFT
GEAR_ROW2_RIGHT
PIN_ROW2_LEFT
PIN_ROW2_RIGHT
SCREW_ROW2_LEFT
SCREW_ROW2_RIGHT
STAND_ROW2_LEFT
STAND_ROW2_RIGHT

BEARING_ROW3_LEFT
BEARING_ROW3_RIGHT
GEAR_ROD_ROW3
GEAR_ROW3_LEFT
GEAR_ROW3_RIGHT
PIN_ROW3_LEFT
PIN_ROW3_RIGHT
SCREW_ROW3_LEFT
SCREW_ROW3_RIGHT
STAND_ROW3_LEFT
STAND_ROW3_RIGHT

BEARING_ROW4_LEFT
BEARING_ROW4_RIGHT
GEAR_ROD_ROW4
GEAR_ROW4_LEFT
PIN_ROW4_LEFT
SCREW_ROW4_LEFT
SCREW_ROW4_RIGHT
STAND_ROW4_LEFT
STAND_ROW4_RIGHT
```

### Produced Subassembly Names

```text
BEARING_STAND_ROW1_RIGHT_ASSEMBLY
GEAR_ROD_ROW1_ASSEMBLY
BEARING_STAND_ROW1_LEFT_ASSEMBLY
FASTENED_STAND_ROW1_RIGHT_ASSEMBLY
UNFASTENED_SECOND_STAND_ROW1_ASSEMBLY
MOUNTED_ROW1_ASSEMBLY
CRANK_MOUNTED_ROW1_ASSEMBLY

BEARING_STAND_ROW2_RIGHT_ASSEMBLY
GEAR_ROD_ROW2_ASSEMBLY
BEARING_STAND_ROW2_LEFT_ASSEMBLY
FASTENED_STAND_ROW2_RIGHT_ASSEMBLY
UNFASTENED_SECOND_STAND_ROW2_ASSEMBLY
MOUNTED_ROW2_ASSEMBLY

BEARING_STAND_ROW3_RIGHT_ASSEMBLY
GEAR_ROD_ROW3_ASSEMBLY
BEARING_STAND_ROW3_LEFT_ASSEMBLY
FASTENED_STAND_ROW3_RIGHT_ASSEMBLY
UNFASTENED_SECOND_STAND_ROW3_ASSEMBLY
MOUNTED_ROW3_ASSEMBLY

BEARING_STAND_ROW4_RIGHT_ASSEMBLY
GEAR_ROD_ROW4_ASSEMBLY
BEARING_STAND_ROW4_LEFT_ASSEMBLY
FASTENED_STAND_ROW4_RIGHT_ASSEMBLY
UNFASTENED_SECOND_STAND_ROW4_ASSEMBLY
MOUNTED_ROW4_ASSEMBLY

COMPLETED_GEARBOX_ASSEMBLY
```

## Completion Criteria

The gearbox assembly is complete when:

- All eight bearings are installed in their corresponding stands.
- All gears are secured to the correct gear rods with wooden pins.
- All four assembled gear rods are supported by their corresponding stand
  pairs.
- All eight stands are secured to `BASE_BOARD`.
- `CRANK_HANDLE_ROW1` is attached to `GEAR_ROD_ROW1`.
- All gears rotate freely and mesh correctly with the adjacent gears.

## Guidance for VLM Answers

- Treat the injected live state as authoritative for current progress and
  readiness. Recommend only a step explicitly listed as `READY`.
- If asked for one next step when several are `READY`, use the same deterministic
  policy as the interface: prefer the row in which a step was most recently
  completed or reverted, then choose the lowest user-facing stage number within
  that row. When that row has no `READY` work, or no row has been worked yet,
  choose the lowest-numbered `READY` row and stage. Treat the global Stage 8
  finish step as last. Explain that this selects a preferred option; other
  listed `READY` steps remain valid and may be performed in any order allowed
  by the graph.
- A step not listed as `COMPLETED` or `READY` is `BLOCKED`. When the selected-step
  context supplies `Blocked by`, repeat those missing prerequisites exactly. If
  that detail is absent, do not guess which input is missing.
- When live context shows that a completed step consumed a raw part and produced
  an assembly, explain that transformation using the supplied canonical names.
  Do not claim that a part is currently active unless the live context says so.
- Use familiar operator-facing names in answers, such as “Row 1 right bearing,”
  “right stand,” and “gear rod.” Accept internal canonical names as input, but do
  not expose graph IDs, uppercase canonical labels, filenames, or row.stage
  numbers unless the user explicitly requests debugging details. Resolve spatial
  terms using the fixed gearbox-local LEFT/RIGHT convention.
- Keep answers concise and action-oriented. Include the row, user-facing stage,
  required parts, and expected output when those details answer the question.
- Never infer extra dependencies from prose order, stage numbers, the diagram,
  or visual proximity. The task graph alone determines `READY` and `BLOCKED`.
- If the live state conflicts with this general description, identify the
  inconsistency instead of silently choosing one source.
