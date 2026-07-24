# Gearbox Assembly Task

## Objective

Assemble four gearbox rows onto `BASE_BOARD`. Each row consists of a gear rod,
one or more matching gears, bearings, gear stands, wooden retaining pins, and
mounting screws. Row 1 also includes a crank handle.

Each row is identified by its corresponding color and shape. A component may
only be assembled with other components assigned to the same row. Components
labeled `LEFT` and `RIGHT` must remain on their designated sides.

## Component Inventory

### Row 1 — One Large Gear

- `BEARING_ROW1_LEFT`
- `BEARING_ROW1_RIGHT`
- `CRANK_HANDLE_ROW1`
- `GEAR_ROD_ROW1`
- `GEAR_ROW1_LEFT` — large gear
- `PIN_ROW1_LEFT`
- `PIN_ROW1_RIGHT`
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
   pins.

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

## Required Assembly Sequence for Each Row

1. Insert the left bearing into the left stand.
2. Insert the right bearing into the right stand.
3. Place the required gear or gears onto the gear rod.
4. Secure the gears with the corresponding wooden pins.
5. Fasten the first bearing-and-stand assembly to `BASE_BOARD` using its
   corresponding screw.
6. Insert the assembled gear rod through the bearing in the fastened stand.
7. At the same time, fit the second bearing-and-stand assembly over the other
   end of the gear rod.
8. Fasten the second stand to `BASE_BOARD` using its corresponding screw.
9. For Row 1 only, attach `CRANK_HANDLE_ROW1` after both stands and the rod are in
   position.

## Completion Criteria

The gearbox assembly is complete when:

- All eight bearings are installed in their corresponding stands.
- All gears are secured to the correct gear rods with wooden pins.
- All four assembled gear rods are supported by their corresponding stand
  pairs.
- All eight stands are secured to `BASE_BOARD`.
- `CRANK_HANDLE_ROW1` is attached to `GEAR_ROD_ROW1`.
- All gears rotate freely and mesh correctly with the adjacent gears.

## Recommended Work Order

Complete the gearbox row by row:

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

This row-by-row order avoids interference between in-progress assemblies on
different rows and keeps the workspace organised.
