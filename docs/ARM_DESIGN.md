# Photo-inspired six-joint wheelchair arm

## Current mobile side-case mount — 22 October 2026

The active mobile task now mounts the main arm on a chassis bracket inside the
side case, below/outboard of the armrest. Nominal floor heights are 580 mm at
the mount and 740 mm at the shoulder; the lateral offset is 390 mm and forward
offset 20 mm. Link lengths, six revolute joints and their limits are unchanged.
See [SIDE_CASE_MOUNT.md](SIDE_CASE_MOUNT.md) for the fitted folded envelope,
updated task paths and measured acceptance evidence.

## Earlier stationary reference design

The dimensions and appearance checks below describe the retained stationary
fixture. Its earlier 721/881 mm heights and 215 mm lateral offset are not the
current mobile mounting dimensions.

The user's supplied side-view drawing is the shape reference: pedestal/body, bicep, forearm and hand. Its printed dimensions are not copied. A single side view does not establish all six joint axes; this design retains the project's six-joint chain rather than claiming an exact Rhino XR-3 reproduction.

## Selected wheelchair-scale dimensions

| Dimension | Value | Reason |
| --- | ---: | --- |
| Upper arm, shoulder to elbow | 300 mm | Existing link span reaches the tested table without extending the arm further |
| Forearm, elbow to wrist centre | 250 mm | Retains the collision-checked reach and lift path |
| Wrist centre to grasp centre | 195 mm | Existing 120 mm flange/end offset plus 75 mm jaw-centre offset |
| Pedestal mount to shoulder | 160 mm | Compact 40 mm mount/waist offset plus 120 mm shoulder riser |
| Nominal mounting height above floor | 721 mm | Existing wheelchair-side mounting position |
| Nominal shoulder height above floor | 881 mm | Tested stationary fixture geometry |
| Side mounting offset from centreline | 215 mm | Existing lateral mount outside the passenger clearance proxy |
| Ideal straight shoulder-to-grasp reach | 745 mm | Sum of rigid link/tool spans; not a guaranteed collision-free working radius |

These dimensions are retained because the existing nominal reach/grasp scene already validates them. The task did not require changing successful link lengths to match the photograph. Suitability here means fit and reach in the tested wheelchair/table simulation; bracket strength, real passenger anthropometry, payload ratings and usability across wheelchair types remain unvalidated.

## Six arm joints, separate finger motion

| Joint | Function | Axis in its local frame | Existing range |
| --- | --- | --- | --- |
| Waist | Base yaw | +Z | -0.6 to 2.6 rad |
| Shoulder | Shoulder pitch | -Y | -1.5 to 2.5 rad |
| Elbow | Elbow pitch | -Y | 0 to 3 rad |
| Wrist 1 | Wrist roll | +X | -2.9 to 2.9 rad |
| Wrist 2 | Wrist pitch | -Y | -2 to 2 rad |
| Wrist 3 | Tool roll | +X | approximately -π to +π |

The two sliding primary jaws are additional tool DOFs. They are not counted toward the six arm joints. The auxiliary arm remains 3-DOF. Joint conventions and allowed bending directions are retained; the reference image's exact depicted pose is not asserted to match them.

## Implemented appearance

`src/arm_appearance.py` is the sole definition of the new style. It modifies only visual geoms/assets in the model tree:

- A rectangular compact pedestal and waist column replace the round visual columns.
- Paired tapered plates replace the bare cylindrical upper-arm and forearm visuals.
- Dark shoulder/elbow covers make the joint locations clear.
- Dark wrist housings and a rectangular tool stem give a compact mechanical hand appearance.
- A light-metal/dark-housing palette distinguishes links from joints. The active grasp fixture retains its actuated jaws.

The paired upper-arm plates have local centres at y=±24 mm, half thickness 4 mm and maximum height ±20 mm. Their radial bound is `sqrt(28^2+20^2)=34.41 mm`, inside the existing 35 mm collision cylinder. Forearm plates have centres ±18 mm, half thickness 3 mm and maximum height ±16 mm; `sqrt(21^2+16^2)=26.40 mm`, inside the existing 27 mm cylinder. Their longitudinal vertices stay inside the original link spans.

The compact pedestal boxes fit inside their original cylinders. New shoulder/elbow visual covers fit inside the existing joint spheres. The tool stem's 7×7 mm half-section fits inside the original 10 mm radius proxy. These conservative proxies intentionally fill the gaps between paired plates: this is not detailed CAD collision geometry. The shells are a visual representation of the existing coarse link mass model; no new shell mass is silently added.

Body transforms, joint axes/limits, explicit masses/inertias, collision geometry/masks, contact profiles, actuator limits and controllers are retained. The canonical mobile XML is unchanged. The active grasp fixture defaults to `arm_appearance="reference"`; `"baseline"` remains available for comparison. `load_model()` retains its baseline default for the deferred mobility work, with the same optional reference style available explicitly. This is an appearance update of one model, not another independently maintained robot XML.

## Verification and preview

Tests compare all body masses/inertias/transforms, joint parameters, equality data, actuator limits and every collision-enabled geom between appearances. Another test transforms the actual compiled mesh vertices into the collision-cylinder frames and checks containment. Existing NumPy FK/Jacobian/pose IK, collision rejection, external wrist obstacle and grasp physics tests remain active.

The full six-case grasp comparison is rerun with the new appearance. Nominal 0.3 kg grasp/lift/hold still passes; the 0.6 kg payload, no-payload-compensation, soft-contact, zero-friction and open-jaw cases retain their documented failures. An appearance change does not resolve those limitations. The stationary world-welded base and other fixture assumptions remain as documented in `GRASP_VALIDATION.md`; mobile stability is not re-certified by this work.

The notebook displays a rendered before/after view on the actual wheelchair plus the grasp animations. Exported previews and measured dimensions are saved to:

- `results/arm_design/arm_comparison.png`
- `results/arm_design/reference_detail.png`
- `results/arm_design/dimensions.json`

Run from the project root:

```sh
.venv/bin/python -m src.preview_arm
.venv/bin/python -m src.run_grasp --compare --render
.venv/bin/python -m unittest discover -s tests -p test_grasp.py -v
.venv/bin/python scripts/check_notebooks.py
```

In VS Code, open `notebooks/wheelchair_current.ipynb`, select **Wheelchair (.venv)**, restart and Run All. Keep `WORKFLOW="grasp"` while mobility is set aside.

Completed appearance verification: all 13 grasp tests pass. The active notebook now also includes the eight transfer tests, comparative graphs and continuous handover animations; current execution results are in `results/notebook_checks/summary.json`. The six physical comparison results retain their prior pass/failure status. The before/after preview uses the same actual arm posture and camera for both styles.
