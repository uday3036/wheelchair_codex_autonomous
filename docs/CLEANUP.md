# Consolidation and cleanup

The single active notebook is `notebooks/wheelchair_current.ipynb`. It imports the canonical model/controller, runs the comparisons, displays quantitative results and representative MuJoCo animations, and runs the regression suite. It keeps the platform/kernel/path/framebuffer fixes and explicit robotics limitations. Subsequent steering work extends this same notebook; see `STEERING_VALIDATION.md`.

The notebook was executed successfully before removing superseded files. The final cleaned layout is also checked with `scripts/check_notebooks.py`. Executed copies and the pass/fail summary are under `results/notebook_checks/`; the current experiment data and visualizations are under `results/current/`.

## Preserved in one recovery archive

`references/history.zip` contains 43 files, verified against the originals byte for byte before deletion from the active tree:

- The five previous project notebooks, including the Milestone 1E controller reference and the NumPy coursework algorithms/experiments.
- The earlier 1C model/notebook and the rigid diagnostic model.
- Professor example notebooks and the reference-only ROS2/Gazebo sample.
- The original progress audit's code, quantitative logs, plots and animations.

The controller-equivalence regression reads the archived Milestone 1E cell. Do not delete the archive without replacing that test reference and preserving the coursework algorithms elsewhere. The fixed-base coursework material remains reference-only with its known collision/planning/control limitations; it was not silently presented as a validated mobile manipulation system.

To recover a historical file without disturbing the active tree, open the ZIP in an archive viewer or extract selected entries to a separate directory. Archive entries retain their original relative paths.

## Removed as redundant

- Extracted copies of the archived sources and reference directories.
- Generated outputs and executed copies for the five superseded notebooks.
- The duplicate `results/contact_validation/` experiment run; the consolidated notebook generates its current equivalent under `results/current/`.
- Redundant requirements files; `requirements.txt` now contains the direct simulation pins and notebook dependencies.
- Placeholder files in nonempty directories, Python bytecode caches, and the stale VS Code Gazebo/CMake setting.

The canonical rocker XML, source modules, physics tests, course requirements, engineering instructions, mathematical/validation documentation and working `.venv` were retained. Source controller gains, physics geometry and numerical acceptance thresholds were not changed during consolidation. The optional spring setting remains explicit; original XML springs remain 45 N m/rad.

At this initial cleanup, generated results were ignored by Git. The later GitHub
publication below selectively includes the evidence required for replay and checks.

## Validated simplification cleanup — 21 October 2026

Archived 215 superseded development files (136.30 MB before compression) in `references/development_evidence_20261021.zip`. Every archived file was verified by SHA-256 against its original before removal. The archive contains `archive_checksums.json`; `results/project_validation/cleanup.json` records its checksum. Earlier failed trials remain available in this archive.

The removed development roots were `results/stowage_development`, `results/autonomous_navigation_checks`, `results/robustness_development`, `results/robustness_candidate` and the old root `MUJOCO_LOG.TXT`. Current quantitative reports, animations, canonical source/model, 146 active mesh pairs, notebook, tests and `.venv` remain available. Seventeen obsolete generated mesh pairs were separately removed only after their hashes matched the prior export manifest.

The original README was preserved in `references/history.zip` as `README_before_simplification.md`. The complete regression run passed 103 tests in 272.379 seconds; its log and source fingerprint are under `results/project_validation/`. The notebook passed with 23 code cells and six image outputs in 328.2 seconds.

## Side-case revision cleanup — 22 October 2026

Archived 181 superseded development/comparison files (227.14 MB
before compression) in `references/side_case_development_20261022.zip`. Every
archived file was verified against its original by SHA-256 before removal. This
includes failed trials, the preceding mount's variation suite, earlier accepted
side-case revisions and pre-timestamp-fix numerical traces. Canonical current
comparisons remain in `results/autonomous_validation/`, `results/full_task_variations/`,
`results/robustness_validation/` and `results/live_validation/`. Source and assets
remain in their established locations.

Three byte-identical native render copies were removed after verifying that the
canonical render files were retained in `results/autonomous_validation/`. Native
raw poses, measured times, metrics and live run status remain available. See
`results/project_validation/side_case_cleanup.json` for checksums and sizes.

## GitHub publication — 29 October 2026

The `.gitignore` now includes the current measured suites, component comparison
evidence, design figures and regression logs needed by the report, replay and
tests. `results/README.md` lists the published evidence. URDF/MJCF/OBJ/STL assets,
the active notebook, complete source and the course report PDF/HTML are retained.

The virtual environment, operating-system/font caches, executed-notebook caches,
duplicate development outputs, generated GIF/video files and large development
ZIP archives remain local. The required `references/history.zip` is included for
the controller-equivalence regression. This publication selection removes no
original local data and does not alter controller/model source or measured gates.
