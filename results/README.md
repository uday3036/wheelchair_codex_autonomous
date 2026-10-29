# Published simulation evidence

The repository includes the measured data, figures and saved poses required for
the default replay, technical report and evidence audits. These files are saved
simulation results, not proof that a new run was executed on the reader's machine.
Compare the source fingerprints in each report before accepting old evidence.

| Directory | Contents |
|---|---|
| `autonomous_validation/` | Nominal complete task and seven baseline expected outcomes |
| `full_task_variations/` | Complete-task pose/noise challenges and obstacle-stop control |
| `robustness_validation/` | Passive-caster reverse and stationary payload/material comparisons |
| `live_validation/` | Verified native live-controller task |
| `display_validation/` | Separate-renderer timing, tests and exact physical-trace comparison |
| `project_validation/` | Historical full regression log and current integration audit |
| `design_validation/`, `side_mount_validation/`, `asset_validation/` | Design figures and model/export checks |
| `grasp_validation/`, `transfer_validation/`, `delivery_validation/` | Stationary component experiments and negative controls |
| `current/` | Retained diagnostic comparison outputs used during development |

Large rendered GIFs/videos, duplicate development runs and executed-notebook
caches remain local and are ignored by Git. The complete nominal motion can be
viewed directly from the included pose recording:

```sh
.venv/bin/mjpython scripts/replay_autonomous.py
```

Regenerate the full nominal animation and baseline comparison figures with:

```sh
.venv/bin/python -B -m src.run_autonomous --acknowledge-passenger --compare --render
```

The large development archives named in older cleanup reports remain in the
original local workspace; they are not required for current simulation/replay.
`references/history.zip` is included because the controller-equivalence
regression reads its original milestone notebook. Historical result folders
omitted here can be regenerated using their documented experiment commands.
