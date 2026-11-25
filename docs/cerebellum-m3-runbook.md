# AlohaMini2 Pro · M3 自由导航与绕椅验收

适用于 5090、双摄树莓派和已签核标定。下列命令在 5090 的 Bash 执行。本次部署目录为 `~/workspace/jjy/validation/m3_4-20261125`。所有操作终端使用同一个 ROS 域、SSH 控制套接字和服务日志目录。

## 连接与现场准备

确认双臂收拢、有支撑，升降下降路径净空。操作员留在物理急停旁。原生 Host 如已有实例，沿用该实例。首次启动会释放双臂力矩并让升降触底回零，现场确认后启动。工作站与树莓派使用有线直连。

```bash
cd ~/workspace/jjy/validation/m3_4-20261125
export ROS_DOMAIN_ID=181 ROS_LOCALHOST_ONLY=1
export M3_NATIVE_REPO="$HOME/workspace/jjy/code/Alohamini_software-lightnav"
export M3_PI_HOST=192.168.2.2 M3_SSH_SOCKET=/tmp/jjy-m34-hw
export M3_RUN_DIR="$PWD/services"
ssh -S "$M3_SSH_SOCKET" -O check pi5@"$M3_PI_HOST" || ssh -MNf -S "$M3_SSH_SOCKET" -o ControlPersist=2h -o StrictHostKeyChecking=yes pi5@"$M3_PI_HOST"
```

若 Host 未启动，在单独终端运行并保持会话：

```bash
ssh -t -S "$M3_SSH_SOCKET" pi5@"$M3_PI_HOST" '~/miniconda3/envs/alohamini_host/bin/alohamini host --robot_model alohamini2pro --bind_host 192.168.2.2 --cameras forward chest'
```

## 默认自由导航

树莓派 Host 已启动时，在 5090 上一行执行自由导航。脚本会检查 SSH、启动模型并显示本次指令。确认路线净空且可以物理急停后输入 `GO`。

```bash
bash ~/workspace/jjy/validation/m3_4-20261125/examples/navigation/m3_navigation.sh run "Follow the corridor and stop at the doorway."
```

加载模型时按 `Ctrl-C` 会撤销本次启动；行驶中按 `Ctrl-C` 会急停、禁用桥并收尾。每轮的结果、图像和 rosbag 在部署目录的 `trials/free-*` 下。如果已有另一个导航实例，脚本会提示占用。按步骤分别操作时可使用以下命令。

```bash
nav=examples/navigation/m3_navigation.sh
unset M3_COLOR M3_TARGET_ID
M3_INSTRUCTION="Follow the open corridor and stop at the doorway." bash "$nav" start
bash "$nav" drive "trials/free-$(date +%Y%m%d-%H%M%S)"
bash "$nav" stop
```

前视 LightNav-0 发布航点和 STOP，双摄持续更新目标约 10 Hz 的局部占据图。空地可直接前进，遇障碍由小脑选择可行方向。自由模式只做系统就绪检查，全程检查双摄、米制地图、轮速、升降、里程计、控制器、桥和速度发布者。可单独执行 `bash "$nav" preflight "trials/static-$(date +%s)"` 做只读核验。

需要颜色目标近场防推停靠时，将示例中的启动行换成：

```bash
M3_COLOR=purple M3_INSTRUCTION="Find the purple toy and stop in front of it." bash "$nav" start
```

可选颜色为 `purple`、`blue`、`green`。有颜色目标也允许从目标尚不可见处起步。新鲜确认的目标进入 0.20 m 前缘净距时，小脑主动制动并核验 ARRIVED。没有有效目标测距时，模型 STOP 经防抖、制动和轮速静止核验后结束本轮，`result.json` 记录 `outcome=model_stop`。已确认的远处目标沿用重观察与新参考恢复流程。换指令或颜色前先执行 `stop`，再按新配置启动。

## 显式绕椅验收

办公椅置于车头正前方 0.8–1.0 m，玩具放在椅子右后方 1.4–1.8 m、从前视可见的位置。右侧至少留 1 m。确认先右移约 0.6–0.9 m、再向前约 1.2–1.5 m 的路线净空。

```bash
nav=examples/navigation/m3_navigation.sh
run_id=$(date +%Y%m%d-%H%M%S)
bash "$nav" start --chair
bash "$nav" preflight "trials/chair-static-$run_id" --chair
bash "$nav" drive "trials/chair-drive-$run_id" --chair
bash "$nav" stop
```

`start --chair` 默认配置紫色玩具；`drive DIR --chair` 才启用专项门禁，要求椅子入图、目标可见、最近 3 秒至少 30 个右绕周期，并在运动中检查专项位移、用时和目标跟踪。验收记录包括对外 ARRIVED、停稳轮速、桥禁用应答及现场零触碰确认。

## 参数

| 项目 | 本次设置及口径 |
|---|---|
| 实机配置 | `evidence/m3-low-speed-shared.json`，建图、测距与小脑共用。覆盖路径使用绝对路径 `M3_CONFIG`，启动时记录 SHA-256。|
| 速度与制动 | 前进/横移/倒退 0.12/0.10/0.06 m/s，合线速度 0.12 m/s，角速度 0.25 rad/s，制动估计 0.45 m/s²。|
| 几何 | 收臂轮廓 x 为 −0.26 至 +0.28 m，y 为 ±0.29 m。到位阈值 0.20 m 按前缘净距计算，正前方相当于距底盘中心 0.48 m。|
| 共享卡时效 | 参考 2.0 s、地图 0.8 s、目标观测 0.5 s。监督器另核验地图接收间隔 0.65 s。独享卡应以实测采集到推理返回的时延和相邻参考到达间隔确定 TTL，再重做静态验收。|
| 双摄建图范围 | 共享配置 `rgb_grid.chest_max_range_m=1.15`。胸部只向距底盘中心水平半径 ≤ 1.15 m 的格子投占据票和空闲票。前视覆盖全射程。地面拟合仍使用完整深度图，独立目标测距沿用原有范围。|
| 目标 | 自由模式默认仅使用语言指令。设置 `M3_COLOR` 后启用米制目标定位，可用 `M3_TARGET_ID` 指定消息标识。|
| 近场与绕行 | 已启用 `metric_arrival_autostop`、`metric_target_detour`。UNKNOWN 计入代价，实占据格及刹车扫掠仍执行硬拒绝。测量模式地面拟合倾角上限 12°。|

## 急停与收尾

现场有风险时立即使用物理急停。软件急停在另一个已设置相同环境的 5090 终端执行：

```bash
bash examples/navigation/m3_navigation.sh estop
```

脚本向 `/alohamini_cerebellum_dry_run/estop` 发布 Bool true 并请求原生桥禁用。手工使用 ROS CLI 时先激活 `~/workspace/jjy/envs/lightnav-ros` 并设置同一 ROS 域：

```bash
ros2 topic pub --once /alohamini_cerebellum_dry_run/estop std_msgs/msg/Bool '{data: true}'
ros2 service call /alohamini_lerobot_bridge/command_enable std_srvs/srv/SetBool '{data: false}'
```

确认车轮停稳、禁用服务返回 success=True。急停锁存后在桥禁用状态下排除原因，重启小脑并重新静态核验。若通信中断直接使用物理急停。结束后运行：

```bash
bash examples/navigation/m3_navigation.sh stop
bash examples/navigation/m3_navigation.sh status
```

`stop` 核对禁用应答后按 PID、启动时刻和工作目录清理本次服务。保留 Host 的停机与扭矩处置由现场操作员在双臂支撑后处理。每次试验使用新输出目录。归档 bag、trace、result.json、双摄起止图、配置哈希及服务日志后再启动下一次会话。
