# Codex repository instructions

## Goal
Develop a simulated autonomous dual-arm assistive wheelchair in MuJoCo using student-written NumPy robotics algorithms. See docs/PROJECT_CONTEXT.md first.

## Required engineering discipline
- Never claim a robotics model works without running the simulation and comparing outputs to acceptance criteria.
- Keep a single canonical model and controller; retain milestone notebooks/models only as references until an explicitly tested replacement is approved.
- Preserve the 6-DOF main manipulator and 3-DOF auxiliary manipulator.
- Mobility intent: two independently driven rear wheels, two actively steered front wheels with passive rolling, two passive swiveling middle casters.
- Validate contact geometry, steer actuator assignments, wheel speed/traction, ground forces, and geometric consistency before blindly tuning gains.
- Preserve animated MuJoCo visualizations and quantitative graphs for meaningful experiments.
- Include tests, logging, mathematical documentation, and run instructions for each changed behavior.
- Coursework core algorithms (FK, Jacobian, IK, path planning, trajectory generation and feedback control) must be implemented by students with NumPy rather than outsourced to built-in solvers; built-ins may verify.
- Prior ROS2/Gazebo sample files are *reference only*: do not substitute Gazebo physics for MuJoCo.
- Make minimal changes, explain expected numerical outcomes, run tests, and state limitations or dependencies if testing is blocked.
- Do not do a wholesale rewrite or claim successful end-to-end bottle opening unless physically validated.
- Keep safety limits and passenger handover clearance constraints explicit in simulation.
