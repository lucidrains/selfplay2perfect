# Play2Perfect assets

Everything simulation-facing lives here. No generated XML or cached meshes.

```
robot/                 KUKA iiwa14 + left Sharpa URDF + STL meshes
table/                 paper table_narrow.urdf
tools/                 cooked AutoBio pipette OBJ (body + tube) and the raw OBJ
beam/                  Fabrica beam_3x parts 0 / 2 / 6 + insertion fixtures
tight_insertion/       L-peg + 0.5 mm hole fixture
furniture/             FurnitureBench square table + leg4
```

Provenance: Play2Perfect / Fabrica / FurnitureBench assets are from the paper's
MIT-licensed release; the pipette OBJ is from AutoBio (check upstream terms).
The MJCF conversion cache is written to `<repo>/.cache/play2perfect/`, not here.

The tools in `tools/` are the only real mesh tool. The paper's hammer,
screwdriver, marker, spatula, eraser and brush are procedural primitives with
the paper's size/mass distributions.
