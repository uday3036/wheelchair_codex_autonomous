# AutoCAD design export

Open `wheelchair_folded_mm.dxf` with AutoCAD's File > Open (DXF file type).
Use Zoom Extents, select an isometric view with the ViewCube, and select a
shaded visual style to inspect the assembly. Use Save As to save a DWG copy.

The drawing contains coloured 3DFACE entities on named component layers.
Units are millimetres (the source OBJ coordinates were multiplied by 1000).
The pose is folded, in the existing base_link coordinate frame; negative
heights below that frame are intentional. Layers can be hidden to inspect
internal parts. No OBJ/MTL sidecar is required to open the DXF.

This is the existing simulation visual assembly, not a parametric CAD solid,
manufacturing drawing, articulated mechanism or fabrication-ready design.
Parts may overlap as in the source visual assembly. See
../../hardware/DESIGN_REVIEW.md for the mechanical design scope.

Regenerate from the project root after exporting the robot assets:

```sh
.venv/bin/python -m pip install ezdxf==1.4.2
.venv/bin/python scripts/export_autocad.py
```

The exporter audits and reloads the DXF and compares every triangle coordinate
with the source after unit conversion. `export_validation.json` records hashes,
bounds and counts. AutoCAD itself has not been run in this environment.
