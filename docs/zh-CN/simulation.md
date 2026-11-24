# 仿真

[English](../simulation.md) | **简体中文**

先完成[仿真环境安装](install.md#4-仿真)。以下命令在仓库根目录、对应环境中执行。

## MuJoCo

打开整机并使用 EGL 设备 0 渲染：

```bash
conda activate alohamini
alohamini-mujoco view --window-egl-device 0
```

省略 `--window-egl-device 0` 使用桌面默认 OpenGL 设备。机器人资产随平台安装，默认固定底盘。

指定升降初值并暂停启动：

```bash
alohamini-mujoco view --window-egl-device 0 \
  --initial-positions '{"vertical_move":0.2}' --paused
```

`--initial-positions` 使用 URDF 关节名；旋转关节单位 rad，升降单位 m，范围为 `[-0.3, 0.3]`。

打开黑色桌面与方块场景；修改 JSON 可调整物体和机器人初态：

```bash
alohamini-mujoco view --window-egl-device 0 \
  --scene-config examples/simulation/research.json --paused
```

## 窗口操作

展开右侧 `Control`，拖动滑块设置关节目标。暂停时保存目标，按空格播放后模型开始跟随。`Joint` 显示当前关节位置；拖动它也会设置对应目标。运动受关节限位、速度和电机力限制。

| 操作 | 效果 |
| --- | --- |
| 空格 | 开始或暂停仿真 |
| 右方向键 | 暂停时推进一个物理步 |
| F9 | 复位并暂停 |
| F8 | 切换视觉模型与碰撞几何 |
| T | 切换半透明显示 |
| F6 | 切换坐标系显示 |
| Tab／Shift + Tab | 显示或隐藏左／右面板 |
| 左键拖动 | 旋转视角 |
| 右键拖动／Shift + 右键拖动 | 在竖直面／水平面平移视角 |
| 滚轮 | 缩放视角 |
| 右键双击 | 将观察中心移到点击位置 |

普通 `view` 中，左键双击选中连杆，Ctrl + 右键拖动施加平移外力，Ctrl + 左键拖动施加转矩；Ctrl + Shift + 右键拖动在水平面施力。

`alohamini-mujoco` 窗口的 `Print data`、模型导出和截图默认保存到 `~/Alohamini_workspace/simulation/viewer/`；自定义工作区使用 `ALOHAMINI_WORKSPACE`。命令中的相对路径按启动目录解析。

## 舵机模型与补偿

有实测舵机参数时，用 `--actuator-profile` 加载 JSON 文件：

```bash
alohamini-mujoco view --actuator-profile /path/to/actuators.json --paused
```

参数文件使用 `alohamini-actuators-v1` 格式，包含 `robot_model`、按 URDF 关节名索引的 `joints` 和记录测量条件、适用范围的 `provenance`。各旋转关节参数为：

| 字段 | 单位 |
| --- | --- |
| `kp_Nm_per_rad`、`kd_Nm_s_per_rad` | N·m/rad、N·m·s/rad |
| `friction_Nm`、`peak_effort_Nm` | N·m |
| `velocity_rad_s`、`acceleration_rad_s2` | rad/s、rad/s² |
| `delay_s`（默认 0） | s |
| `reference_velocity_gain_Nm_s_per_rad`（默认 0） | N·m·s/rad |
| `gravity_position_gain_rad_per_Nm`（可选） | rad/(N·m) |
| `ki_Nm_per_rad_s`（默认 0） | N·m/(rad·s) |
| `integral_effort_limit_Nm`、`integral_deadband_rad` | N·m、rad |
| `integral_period_s`（默认每个物理步更新） | s |

启用 I 时，积分扭矩限幅须大于 0 且不超过峰值扭矩。

`--feedforward auto` 为默认设置：实测关节使用参数文件中的舵机响应，其余关节使用仿真 PD 与重力／科里奥利前馈。`--feedforward gravity` 仅补偿重力；实测关节使用标定的位置前馈增益，省略时使用 `1/kp_Nm_per_rad`。位置前馈受关节行程、速度、加速度和峰值扭矩限制。`none` 关闭全部外部前馈，`bias` 补偿完整动力学偏置。动作 JSON 的 `controller.feedforward` 使用相同取值。

`peak_effort_Nm` 控制瞬时扭矩上限，URDF 的 `effort` 用于连续负载校核。参数须对应实际舵机、PID、供电与安装条件；真机补偿通过实测验证后部署。

## ManiSkill 与 RoboTwin

在 ManiSkill 环境查看同一场景：

```bash
conda activate alohamini
alohamini-maniskill view --scene-config examples/simulation/research.json
```

窗口目标刷新率为 60 FPS，使用 `--render-fps 30` 可调整。控制频率由场景的 `control_frequency_hz` 设置，物理步长由 `timestep_s` 设置。

Linux X11 混合显卡默认使用隔离窗口，保留原生查看器。`--window-mode native` 使用直接窗口，`--window-mode isolated` 强制隔离窗口。

在 RoboTwin 环境查看，`--source` 指向其源码目录：

```bash
conda activate alohamini_robotwin
alohamini-robotwin view --source /path/to/RoboTwin \
  --scene-config examples/simulation/research.json
```

添加 `--headless --steps 1000` 可关闭渲染并推进 1000 个物理步；默认步长为 1 ms。两者当前支持单个 CPU 物理环境。

## 自定义桌面场景

示例使用独立的 JSON 文件定义场景、任务和配套关节轨迹。

```bash
SCENE=examples/simulation/pick_place
EPISODE="$HOME/Alohamini_workspace/simulation/collections/tabletop_01"
alohamini-mujoco view --scene-config "$SCENE/tabletop.json" \
  --task-config "$SCENE/task.json" --trajectory "$SCENE/motion.json" --paused
MUJOCO_GL=egl alohamini-mujoco collect "$SCENE/motion.json" \
  --scene-config "$SCENE/tabletop.json" --task-config "$SCENE/task.json" --output "$EPISODE"
alohamini-mujoco replay "$EPISODE" --view --paused
```

输出目录须为新目录，采集保存 RGB 帧、关节状态与命令，以及供回放使用的模型。要试用加高桌面和另一组物体位置，使用 `tabletop_raised.json` 和 `motion_raised.json`，并换一个输出目录。

用 `T_world_object` 设置物体位姿，用 `T_world_base` 设置底盘位姿。位置和尺寸用 m，质量用 kg。`pick_place` 任务指定可移动的 `source` 和水平放置的固定方盒 `target`，底盘固定，控制频率为 20 Hz。更换物体或机器人位置后，使用与新布局匹配的轨迹。示例使用虚拟工作区相机和腕部相机。自定义场景相机与 `mujoco_contact_time_constant_s` 由 MuJoCo 后端支持。

### 在窗口摆放物体与取景

```bash
alohamini-mujoco edit --scene-config examples/simulation/pick_place/tabletop.json \
  --task-config examples/simulation/pick_place/task.json \
  --output "$HOME/Alohamini_workspace/simulation/tabletop_custom.json"
```

- 双击场景物体。Ctrl + 右键拖动平移，加 Shift 沿水平面移动。Ctrl + 左键拖动旋转。F12 在自由移动和世界 X、Y、Z 单轴移动间切换。
- 在 `Rendering → Camera → Free` 调整视角，按 F11 记录为 `workspace`。用 `--camera-name` 指定另一个相机名。
- 按 F10 校验并保存到新 JSON。添加 `--task-config` 同时检查任务初态。随后将新 JSON 传给 `view` 或 `collect`；采集使用与该布局匹配的轨迹。

### 素材引用与无窗口预览

场景物体可用 `"asset": {"id": "blue_cube", "version": 1}` 引用[目录中的素材](../../examples/simulation/pick_place/assets/catalog.json)，实例位姿仍用 `T_world_object`。目录文件记录三角面 OBJ、可选 UV 对应的 PNG 贴图、显式碰撞体和文件 SHA-256。碰撞体支持盒子、球体、圆柱和凸 OBJ。可移动资产声明质量、质心和中心惯性张量，单位为 kg、m、kg·m²；省略物性表示固定物体。

资源路径相对于目录 JSON；OBJ 默认用 m，毫米网格用 `"scale": 0.001`。实例的 `asset.sha256` 可选，取校验结果中对应 `object_assets` 条目的 `sha256`，用于固定定义内容。

```bash
SCENE=examples/simulation/pick_place
MUJOCO_GL=egl alohamini-mujoco inspect-scene \
  --scene-config "$SCENE/tabletop_asset.json" --asset-catalog "$SCENE/assets/catalog.json" \
  --task-config "$SCENE/task.json" --camera workspace \
  --preview "$HOME/Alohamini_workspace/simulation/workspace.png"
```

命令输出 JSON，并在校验失败时返回退出码 1。省略 `--preview` 只校验；用 `--collisions` 预览碰撞几何。`view`、`edit`、`collect` 使用相同的 `--scene-config` 和 `--asset-catalog`。回放从回合目录读取模型和资源。当前素材实例由 MuJoCo 后端加载。

## LIBERO 场景与回放

按[安装文档](install.md#libero-场景与参考数据)解压配套资产包后，在当前终端设置路径：

```bash
export SIM_ASSETS="$HOME/Alohamini_workspace/simulation/alohamini-mujoco-assets-v1"
export LIBERO_BUNDLE="$SIM_ASSETS/libero_spatial"
export REFERENCE_DATASET="$SIM_ASSETS/reference/task8"
export COLLISION_BUNDLE="$SIM_ASSETS/robot/collision"
export INERTIAL_BUNDLE="$SIM_ASSETS/robot/inertia.json"
```

打开 LIBERO Spatial 任务，`--task-index` 为 0–9，`--reset-index` 为 0–49：

```bash
alohamini-mujoco view --window-egl-device 0 \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --task-index 8 --reset-index 0 --paused
```

回放参考回合，或将路径替换为自己的回合目录：

```bash
alohamini-mujoco replay "$REFERENCE_DATASET" \
  --view --window-egl-device 0 --paused
```

回放使用该回合保存的模型与初态，动作结束自动暂停；省略 `--view` 只验证回放。

## 采集数据

使用参考轨迹与配套机器人资产采集一个回合：

```bash
DATASET="$HOME/Alohamini_workspace/simulation/collections/task8_demo_01"
MUJOCO_GL=egl alohamini-mujoco collect "$REFERENCE_DATASET/trajectory.json" \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --task-index 8 --reset-index 0 --output "$DATASET"
```

输出目录须为新目录。`--cameras` 选择相机，`--image-size 480 640` 设置图像高和宽；默认使用三路 128×128 图像、20 Hz 控制。更换布局或初态后需生成匹配轨迹。

从参考回合批量采集：

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

中断后用原命令加 `--resume`。初态 0–39 用于训练，40–49 用于评估；更换参考回合须使用匹配配方。批次结果写入 `index.json`，训练只读取已收录的训练组回合。训练参数见[训练手册](training.md)。
