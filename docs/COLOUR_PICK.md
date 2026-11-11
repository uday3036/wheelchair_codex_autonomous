# Three coloured bottles and a typed command — 11 November 2026

Three bottles (red, green, blue) stand on the table. A typed command chooses one; the chair finds that
colour with its camera, docks to it and does the full task (grasp, hand to the auxiliary arm, remove
the cap, drop it in the bin, hand the bottle to the passenger). The other two bottles stay on the table.

Code: `src/bottle_command.py` (command parser, layout), `bottle_scene(bottles=..., pick=...)`,
`detect_bottle(..., color=...)`, flags `--pick / --command / --ask / --on-table` in
`src/run_autonomous.py`, `scripts/run_live.py` and `scripts/live_task.py`.

## The command

`parse_command(text)` accepts a sentence only if it names **exactly one** known colour:

| Typed | Result |
| --- | --- |
| `bring me the green bottle`, `pick red`, `BLUE, please!` | accepted: green / red / blue |
| `yellow please`, `bring me a bottle`, empty | refused: no known colour |
| `red or blue` | refused: ambiguous |
| `not the blue one`, `anything except green` | refused: contains a negation |

A refused command exits (code 2) with the reason **before the simulation starts**, so the chair never
moves on a mistyped command. An accepted colour that is not on the table (`--on-table red,blue --pick green`)
is refused by perception: no green bottle is detected, driving and grasping are inhibited (0 m travelled).

## The table

All bottles at the single-bottle task's depth, 0.36 m apart, the middle one straight ahead of the start:

| Bottle | World x, y (m) | Chair docks at y (m) |
| --- | --- | --- |
| red | 0.62, 0.575 | +0.36 |
| green | 0.62, 0.215 | 0 |
| blue | 0.62, −0.145 | −0.36 |

Table top x 0.48–0.88 (as before), y −0.365–0.795 (wider); legs at x 0.82/0.86, at its ends.
The chair always docks with the commanded bottle at the same place relative to itself
(goal = bottle − (0.62, 0.215)), so the arms run the same validated motions. Two rules came from runs:

* **Same depth.** In an earlier staggered layout blue stood 85 mm deeper; the table's near edge was then
  85 mm closer to the chair and the auxiliary arm's hand-off pose went 4.7 mm into it (route refused).
* **0.36 m apart.** The main arm is on the chair's left and the cap bin is 0.185 m to the target's left, so
  that side must be free. At 0.26 m the palm hit the left neighbour carrying the cap (3.7 N, task stopped).

Distractor bottles have their own free joints and colliding cylinders; a distractor resting on the table is
the only contact with it that is legal (`resting_distractor`); any contact with the robot stops the task.
`multiccd` is enabled so a standing cylinder has a stable multi-point footprint. With distractors present the
grasp approach starts 0.12 m above the grasp (instead of 0.04 m) so the hand passes over their tops.

## Results (headless, `--acknowledge-passenger --pick <colour>`, 11 Nov 2026)

| Arm | Colour | Task | Docking error | Delivery error | Passenger clearance | Recipient force | Other bottles moved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| buildable (`--realistic-arm`) | red | passed | 1.97 mm | 3.22 mm | 41.6 mm | 1.46 N | 0.03 mm |
| buildable | green | passed | 1.28 mm | 3.22 mm | 41.8 mm | 1.45 N | 0.03 mm |
| buildable | blue | passed | 3.10 mm | 3.22 mm | 41.5 mm | 1.46 N | 0.03 mm |
| reference | red | passed | 1.80 mm | 3.83 mm | 37.4 mm | 1.42 N | 0.03 mm |
| reference | green | passed | 1.27 mm | 3.83 mm | 37.9 mm | 1.42 N | 0.03 mm |
| reference | blue | passed | 2.70 mm | 3.80 mm | 37.0 mm | 1.42 N | 0.03 mm |

Limits: passenger clearance > 10 mm, recipient force < 6 N, docking < 5 mm. No unexpected contacts in any run;
cap detached, disposed in the bin, and the arms stowed afterwards in all six. About 320 s simulated each.

## Tests

`tests/test_colour_pick.py` (11 tests, about 4 min): parser accept/refuse, refusal exits before simulation,
layout and table margins, task bottle vs distractors, distractors stay still, contact whitelist, every
colour detected and docked within 5 mm, an absent colour inhibits driving, and the full blue task (farthest
sideways dock, neighbour on the arm's side) with the other bottles moving less than 2 mm.

## Run

```sh
.venv/bin/python -B scripts/run_live.py --realistic-arm --command "bring me the green bottle" --acknowledge-passenger --speed 1
.venv/bin/python -B scripts/run_live.py --realistic-arm --ask --acknowledge-passenger --speed 1
.venv/bin/python -B scripts/run_live.py --realistic-arm --pick blue --on-table red,green --speed 1   # refused: no blue
.venv/bin/python -B -m unittest tests.test_colour_pick -v
```

## Limits

* One fixed layout. Bottles closer than 0.36 m sideways, or at different depths, were not validated and the
  cases above show why they fail; the chair does not re-plan the arm motion around arbitrary neighbours.
* Perception is the colour mask on ideal RGB-D (no lighting changes, no sensor noise in these runs); the
  three colours are well separated, similar colours were not tested.
* Sideways docking up to 0.36 m from the start line (planner limit 0.60 m); obstacle routes (`--obstacles`)
  were not combined with the three-bottle table.
* Cap opening is the existing simulated thread model, not physical validation of a real cap.
