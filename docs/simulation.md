# Simulation

**English** | [简体中文](zh-CN/simulation.md)

Complete [simulation environment installation](install.md#4-simulation) first. Run these commands from the repository root in the corresponding environment.

## MuJoCo

Open the full robot with EGL device 0 for rendering:

```bash
conda activate alohamini
alohamini-mujoco view --window-egl-device 0
```

Omit `--window-egl-device 0` to use the desktop's default OpenGL device. Robot assets are installed with the platform; the base is fixed by default.

Set the initial lift position and start paused:

```bash
alohamini-mujoco view --window-egl-device 0 \
  --initial-positions '{"vertical_move":0.2}' --paused
```

`--initial-positions` takes URDF joint names. Revolute joints use rad; the lift uses m in `[-0.3, 0.3]`.

Open the black-table-and-cube scene; edit the JSON to adjust object and robot initial states:

```bash
alohamini-mujoco view --window-egl-device 0 \
  --scene-config examples/simulation/research.json --paused
```

## Window controls

Expand `Control` on the right and drag sliders to set joint targets. Targets are stored while paused; press Space to play and let the model follow. `Joint` shows current joint positions; dragging it also sets the corresponding target. Motion respects joint, velocity, and motor-force limits.

| Input | Effect |
| --- | --- |
| Space | Start or pause simulation |
| Right arrow | Advance one physics step while paused |
| F9 | Reset and pause |
| F8 | Toggle visual models and collision geometry |
| T | Toggle transparency |
| F6 | Toggle frame display |
| Tab / Shift + Tab | Show or hide the left/right panel |
| Left drag | Rotate the view |
| Right drag / Shift + right drag | Pan in the vertical/horizontal plane |
| Scroll wheel | Zoom |
| Right double-click | Move the view center to the clicked point |

In standard `view`, left double-click selects a link; Ctrl + right drag applies translational force, Ctrl + left drag applies torque, and Ctrl + Shift + right drag applies force in the horizontal plane.

The `alohamini-mujoco` window saves `Print data`, model exports, and screenshots under `~/Alohamini_workspace/simulation/viewer/` by default, respecting `ALOHAMINI_WORKSPACE`. Relative command paths resolve from the launch directory.

## Servo models and compensation

Load the bundled AlohaMini2Pro dual-arm profile:

```bash
alohamini-mujoco view --paused \
  --actuator-profile models/alohamini2pro/config/actuators/arm_stock.json
```

Both arms share the right arm's measured response by joint, including the grippers. Measurements used low-speed, small-angle motions with an empty gripper and baseline servo settings. The profile uses the reference speed and acceleration from those tests and each joint's travel limits from the URDF. For other setups, `--actuator-profile` accepts a custom JSON file.

The `alohamini-actuators-v1` format includes `robot_model`, `joints` keyed by URDF joint name, and `provenance` documenting measurement conditions and applicability. Revolute-joint parameters are listed below.

| Field | Unit |
| --- | --- |
| `kp_Nm_per_rad`, `kd_Nm_s_per_rad` | N·m/rad, N·m·s/rad |
| `friction_Nm`, `peak_effort_Nm` | N·m |
| `velocity_rad_s`, `acceleration_rad_s2` | rad/s, rad/s² |
| `delay_s` (default 0) | s |
| `reference_velocity_gain_Nm_s_per_rad` (default 0) | N·m·s/rad |
| `gravity_position_gain_rad_per_Nm` (optional) | rad/(N·m) |
| `ki_Nm_per_rad_s` (default 0) | N·m/(rad·s) |
| `integral_effort_limit_Nm`, `integral_deadband_rad` | N·m, rad |
| `integral_period_s` (default: update every physics step) | s |

With I enabled, integral torque limits must be positive and no greater than peak torque.

`--feedforward auto` is the default: measured joints use the file's servo response, while others use simulated PD with gravity/Coriolis feedforward. `--feedforward gravity` compensates gravity only; measured joints use calibrated position-feedforward gain, falling back to `1/kp_Nm_per_rad` when omitted. Position feedforward respects joint travel, velocity, acceleration, and peak-torque limits. `none` disables all external feedforward; `bias` compensates the full dynamics bias. Action JSON `controller.feedforward` accepts the same values.

`peak_effort_Nm` limits instantaneous torque; URDF `effort` is used for continuous-load checks. Parameters must match actual servos, PID, power supply, and installation conditions. Validate compensation with hardware measurements before deployment.

## ManiSkill and RoboTwin

View the same scene in the ManiSkill environment:

```bash
conda activate alohamini
alohamini-maniskill view --scene-config examples/simulation/research.json
```

The window targets 60 FPS; adjust with `--render-fps 30`. Scene `control_frequency_hz` sets control frequency, and `timestep_s` sets the physics step.

Linux X11 systems with hybrid graphics default to an isolated window, preserving the native viewer. Use `--window-mode native` for a direct window or `--window-mode isolated` to force isolation.

View in the RoboTwin environment, with `--source` pointing to its source directory:

```bash
conda activate alohamini_robotwin
alohamini-robotwin view --source /path/to/RoboTwin \
  --scene-config examples/simulation/research.json
```

Add `--headless --steps 1000` to disable rendering and advance 1000 physics steps; the default step is 1 ms. Both currently support one CPU physics environment.

## Custom tabletop scene

The example defines a scene, a task, and a matching joint trajectory in separate JSON files.

```bash
SCENE=examples/simulation/pick_place
EPISODE="$HOME/Alohamini_workspace/simulation/collections/tabletop_01"
alohamini-mujoco view --scene-config "$SCENE/tabletop.json" \
  --task-config "$SCENE/task.json" --trajectory "$SCENE/motion.json" --paused
MUJOCO_GL=egl alohamini-mujoco collect "$SCENE/motion.json" \
  --scene-config "$SCENE/tabletop.json" --task-config "$SCENE/task.json" --output "$EPISODE"
alohamini-mujoco replay "$EPISODE" --view --paused
```

Use a new episode directory. The collector saves RGB frames, joint states and commands, and the model for replay. To try the raised table and alternate object positions, use `tabletop_raised.json` with `motion_raised.json` and a new output directory.

Set object poses with `T_world_object` and the chassis pose with `T_world_base`. Use metres for positions and object dimensions, and kilograms for mass. The `pick_place` task uses a movable `source` and an upright fixed box `target`, with a fixed robot base and 20 Hz control. Update the matching trajectory when moving the objects or the robot. The supplied workspace and wrist cameras are virtual views. Custom scene cameras and `mujoco_contact_time_constant_s` use the MuJoCo backend.

### Place objects and capture a camera

```bash
alohamini-mujoco edit --scene-config examples/simulation/pick_place/tabletop.json \
  --task-config examples/simulation/pick_place/task.json \
  --output "$HOME/Alohamini_workspace/simulation/tabletop_custom.json"
```

- Double-click a scene object. Ctrl + right-drag moves it, with Shift for the horizontal plane. Ctrl + left-drag rotates it. F12 cycles between free movement and the world X, Y, and Z axes.
- Frame the scene in `Rendering → Camera → Free` and press F11 to capture `workspace`. Use `--camera-name` for a different name.
- Press F10 to validate and save to a new JSON file. With `--task-config`, validation includes the task reset. Pass the new JSON to `view` or `collect`. Use a trajectory that matches the layout when collecting.

### Asset references and headless preview

Scene objects can reference an [asset catalog](../examples/simulation/pick_place/assets/catalog.json) entry with `"asset": {"id": "blue_cube", "version": 1}`. Set the instance pose with `T_world_object`. The catalog declares triangulated OBJ visuals, optional UV-mapped PNG textures, explicit colliders, and file SHA-256 hashes. Colliders can be boxes, spheres, cylinders, or convex OBJ meshes. For dynamic assets, supply mass, centre of mass, and central inertia in kg, m, and kg·m². Omitting mass properties defines a fixed asset.

Resource paths are relative to the catalog JSON. OBJ coordinates default to metres; use `"scale": 0.001` for millimetres. To pin the asset definition, set the optional instance field `asset.sha256` to the corresponding `object_assets` entry's `sha256` from inspection.

```bash
SCENE=examples/simulation/pick_place
MUJOCO_GL=egl alohamini-mujoco inspect-scene \
  --scene-config "$SCENE/tabletop_asset.json" --asset-catalog "$SCENE/assets/catalog.json" \
  --task-config "$SCENE/task.json" --camera workspace \
  --preview "$HOME/Alohamini_workspace/simulation/workspace.png"
```

The command prints JSON and returns exit status 1 on validation failure. Omit `--preview` to validate alone or add `--collisions` to render collision geometry. Use the same `--scene-config` and `--asset-catalog` with `view`, `edit`, or `collect`. Replay reads the model and resources from the episode directory. Catalog object instances use the MuJoCo backend.

## LIBERO scenes and replay

After extracting the asset archive according to [Installation](install.md#libero-scenes-and-reference-data), set its path in the current terminal:

```bash
export SIM_ASSETS="$HOME/Alohamini_workspace/simulation/alohamini-mujoco-assets-v1"
export LIBERO_BUNDLE="$SIM_ASSETS/libero_spatial"
export REFERENCE_DATASET="$SIM_ASSETS/reference/task8"
export COLLISION_BUNDLE="$SIM_ASSETS/robot/collision"
export INERTIAL_BUNDLE="$SIM_ASSETS/robot/inertia.json"
```

Open a LIBERO Spatial task with `--task-index` 0–9 and `--reset-index` 0–49:

```bash
alohamini-mujoco view --window-egl-device 0 \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --task-index 8 --reset-index 0 --paused
```

Replay a reference episode, or replace the path with your own episode directory:

```bash
alohamini-mujoco replay "$REFERENCE_DATASET" \
  --view --window-egl-device 0 --paused
```

Replay uses the episode's saved model and initial state and pauses automatically when actions end. Omit `--view` to validate replay only.

## Collect data

Collect one episode using a reference trajectory and matching robot assets:

```bash
DATASET="$HOME/Alohamini_workspace/simulation/collections/task8_demo_01"
MUJOCO_GL=egl alohamini-mujoco collect "$REFERENCE_DATASET/trajectory.json" \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --task-index 8 --reset-index 0 --output "$DATASET"
```

The output directory must be new. `--cameras` selects cameras; `--image-size 480 640` specifies image height and width. Defaults are three 128×128 streams and 20 Hz control. Changed layouts or initial states require matching trajectories.

Collect a batch from reference episodes:

```bash
DATASET="$HOME/Alohamini_workspace/simulation/collections/task8_batch_01"
MUJOCO_GL=egl alohamini-mujoco collect \
  --reference "$REFERENCE_DATASET" \
  --recipe examples/simulation/libero_motion_forward.json \
  --reset-indices {0..39} --task-index 8 \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --output "$DATASET"
```

After interruption, rerun the same command with `--resume`. Initial states 0–39 are for training and 40–49 for evaluation; changing reference episodes requires a matching recipe. Batch results go to `index.json`; training reads only indexed training-group episodes. See the [training guide](training.md) for parameters.
