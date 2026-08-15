# Colored gearbox STL files

These files preserve the source geometry and encode one uniform color per part
using the VisCAM/SolidView 15-bit binary-STL facet-color convention.

Color assignment:

- `BaseBoard.stl`, `Bearing.stl`: black
- `Handle.stl`: white
- `Row1_*`: white, except `Row1_Screws.stl`: silver
- `Row2*`: red, except `Row2_Screws.stl`: black
- `Row3_*`: green, except `Row3_Screws.stl`: black
- `Row4_*`: blue, except `Row4_Screws.stl`: silver
- `WoodenPin.stl`: brown

`GearboxAssemblyFull.stl` is not copied because a single uniform color would
not represent its differently colored parts. The original STL files in the
parent directory are unchanged.

Note that STL color is a nonstandard extension and some viewers ignore it. A
viewer that supports VisCAM/SolidView binary-STL facet colors is required.
