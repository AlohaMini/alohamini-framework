# AlohaMini2 Pro · M3 低速绕椅停靠

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

## 启动、核验、单次运行

```bash
bash examples/navigation/m3_navigation.sh start
bash examples/navigation/m3_navigation.sh status
run_id=$(date +%Y%m%d-%H%M%S)
bash examples/navigation/m3_navigation.sh preflight "trials/static-$run_id"
```

`start` 先启动原生桥并核实命令已禁用，随后启动时间校正双摄、Metric3D-384 与 LightNav-0、建图和到位节点、低速小脑以及前视导航。模型端口为本机 18642 和 18643。若树莓派时钟漂移，查看 camera.log 的 Pi-PC 区间与 grid.log 两路地图更新数。在桥禁用时校时。等待同步后重启已退出的本轮服务，再做静态核验。

椅子位于车头正前 0.8–1.0 m，紫色玩具在右后方开阔侧，起点可见玩具，右侧净空至少 1 m。静态核验采样 4.5 秒。通过条件包括车轮静止、升降就绪、唯一速度发布者、双摄图像、新鲜有效测距、实测椅子占据和最近 3 秒内至少 30 个右绕决策。结果保存在本次目录的 result.json。改变摆放或配置后重新核验。

操作员确认向右约 0.6–0.9 m，再沿椅子右侧向前约 1.2–1.5 m 的整段路线净空、可随时急停后执行：

```bash
bash examples/navigation/m3_navigation.sh drive "trials/drive-$run_id"
```

核对终端提示并输入 GO。脚本先启动 rosbag，再次执行静态核验，通过后才调用原生桥使能服务。控制器依据独立曝光确认的目标前缘净距主动制动并验证到位。成功记录应同时包含对外 ARRIVED、正常停止的轮速和桥禁用应答。椅脚剐蹭和玩具碰触由现场操作员核验。

## 参数

| 项目 | 本次设置及口径 |
|---|---|
| 实机配置 | `evidence/m3-low-speed-shared.json`，建图、测距与小脑共用。覆盖路径使用绝对路径 `M3_CONFIG`，启动时记录 SHA-256。|
| 速度与制动 | 前进/横移/倒退 0.12/0.10/0.06 m/s，合线速度 0.12 m/s，角速度 0.25 rad/s，制动估计 0.45 m/s²。|
| 几何 | 收臂轮廓 x 为 −0.26 至 +0.28 m，y 为 ±0.29 m。到位阈值 0.20 m 按前缘净距计算，正前方相当于距底盘中心 0.48 m。|
| 共享卡时效 | 参考 2.0 s、地图 0.8 s、目标观测 0.5 s。监督器另核验地图接收间隔 0.65 s。独享卡应以实测采集到推理返回的时延和相邻参考到达间隔确定 TTL，再重做静态验收。|
| 双摄建图范围 | 共享配置 `rgb_grid.chest_max_range_m=1.15`。胸部只向距底盘中心水平半径 ≤ 1.15 m 的格子投占据票和空闲票。前视覆盖全射程。地面拟合仍使用完整深度图，独立目标测距沿用原有范围。|
| 目标 | 默认紫色。颜色由 `M3_COLOR` 指定，`M3_TARGET_ID` 与 `M3_INSTRUCTION` 应描述同一目标，修改后重新启动。|
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
