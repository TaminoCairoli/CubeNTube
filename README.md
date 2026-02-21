# Cube'n Tube — Shape Erasers for ChimeraX

A ChimeraX plugin that extends the built-in map eraser with three additional eraser shapes: **cube**, **cylinder**, and **custom volume**. All three are accessible from a single unified panel with a shape dropdown.

## Features

- **Cube eraser** — axis-independent box with adjustable X/Y/Z dimensions and optional lock to scale uniformly
- **Cylinder eraser** — frustum with independent top/bottom radii and length, optional radius lock
- **Custom eraser** — use any displayed volume isosurface as the eraser shape, with uniform scaling
- **Single-step undo** (Cmd+Z / Ctrl+Z) for all eraser shapes, including the built-in sphere eraser
- Right-click drag to reposition the eraser in the viewport (Shift+drag for depth)
- Erase inside, erase outside, or reduce map bounds to the eraser region

## Requirements

- [UCSF ChimeraX](https://www.rbvi.ucsf.edu/chimerax/) 1.11 or later

## Installation



## Usage

After starting ChimeraX the **Cube'n Tube** button appears in the **Right Mouse > Map** toolbar section.

You can also open the panel from the menu: **Tools > Volume Data > CubeNTube**.


1. Open a density map in ChimeraX
2. Click the **Cube'n Tube** button in the Right Mouse toolbar (or open **Tools > Volume Data > CubeNTube**)
3. Select a shape from the dropdown (Cube, Cylinder, or Custom)
4. Adjust dimensions with the sliders
5. Right-click drag to position the eraser over the region of interest
6. Click **Erase inside** or **Erase outside**
7. Press Cmd+Z to undo if needed (Only one undo / redo at a time)

For the **Custom** eraser, first display an isosurface on the volume you want to use as a mask, select it from the dropdown, and click **Set as eraser**.

## Project Structure

```
bundle_info.xml        — Bundle configuration (tools, commands, toolbar button)
src/
  __init__.py          — Entry point: registers mouse modes and patches sphere undo
  cube_eraser.py       — Cube model, erase math, mouse mode, standalone panel
  cylinder_eraser.py   — Cylinder model, erase math, mouse mode, standalone panel
  custom_eraser.py     — Custom volume model, erase math, mouse mode, standalone panel
  shape_eraser.py      — CubeNTube unified panel with shape dropdown (QComboBox + QStackedWidget)
```

## Acknowledgements

This plugin was developed with AI assistance (Claude / Cursor). The core erase logic follows the same coordinate-transform approach used by ChimeraX's built-in sphere eraser.

ChimeraX is developed by the [UCSF Resource for Biocomputing, Visualization, and Informatics](https://www.rbvi.ucsf.edu/chimerax/).

## License

This project is open source, licensed under the [GNU Lesser General Public License v2.1](LICENSE). You are free to use, modify, and redistribute this code.
