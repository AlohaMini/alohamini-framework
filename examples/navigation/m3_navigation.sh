#!/usr/bin/env bash
# M3 AlohaMini2 Pro operator entry point, on the inference workstation.
set -eo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
jjy=${JJY_HOME:-"$HOME/workspace/jjy"}
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${M3_ROS_ENV:-$jjy/envs/lightnav-ros}"
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-181} ROS_LOCALHOST_ONLY=1
[[ "$ROS_DOMAIN_ID" =~ ^[1-9][0-9]*$ ]] || { echo "Set an isolated ROS domain." >&2; exit 2; }
native=${M3_NATIVE_REPO:-"$repo"}
export PYTHONPATH="$repo/src:$repo/ros2/src/alohamini_bridge:$native/ros2/src/alohamini_camera:$repo/examples/navigation"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export M3_CONFIG=${M3_CONFIG:-"$repo/evidence/m3-low-speed-shared.json"}
color=${M3_COLOR:-purple}
export M3_TARGET_ID=${M3_TARGET_ID:-"${color}_toy"}
instruction=${M3_INSTRUCTION:-"Approach the $color toy and stop in front of it."}
host=${M3_PI_HOST:-192.168.2.2}
peer=${M3_PI_USER:-pi5}@$host
socket=${M3_SSH_SOCKET:-"/tmp/jjy-m3-$UID-$ROS_DOMAIN_ID"}
export M2B_OPENVO=${M2B_OPENVO:-"$HOME/xyc-auto/new_bases_round2_20261121/openvo"}
export M2B_LIGHTNAV_MODEL=${M2B_LIGHTNAV_MODEL:-"$jjy/models/LightNav-0"}
export M2B_GPU_ENV=${M2B_GPU_ENV:-"$jjy/envs/lightnav-server"}
export M2B_DEPTH_WIDTH=384
run=${M3_RUN_DIR:-"$jjy/logs/m3-navigation-$ROS_DOMAIN_ID"}
mkdir -p "$run"
service=/alohamini_lerobot_bridge/command_enable
ctrl=/alohamini_cerebellum_dry_run
launch() {
    local role=$1
    shift
    nohup setsid "$@" </dev/null >"$run/$role.log" 2>&1 8>&- &
    local pid=$!
    echo "$pid $(awk '{print $22}' /proc/$pid/stat)" >"$run/$role.pid"
}
owned() {
    local pid birth
    [[ -r "$run/$1.pid" ]] || return 1
    read -r pid birth <"$run/$1.pid"
    [[ "$pid" =~ ^[0-9]+$ && -r /proc/$pid/stat && "$(readlink /proc/$pid/cwd)" == "$repo" &&
       "$(awk '{print $22}' /proc/$pid/stat)" == "$birth" ]]
}
disable() {
    local reply
    reply=$(timeout 8 ros2 service call "$service" std_srvs/srv/SetBool '{data: false}') || return 1
    echo "$reply"
    [[ "$reply" == *"success=True"* ]]
}
cleanup() {
    local role pid
    for role in brain controller arrival grid services camera bridge; do
        if owned "$role"; then read -r pid _ <"$run/$role.pid"; kill -TERM "$pid" || true; fi
    done
}
action=${1:-status}
case "$action" in
start)
    exec 8>"$run/start.lock"; flock -n 8 || exit 1
    for role in bridge services controller camera grid arrival brain; do
        if owned "$role"; then echo "$role is already running in $run" >&2; exit 1; fi
    done
    [[ "$M3_CONFIG" = /* && -f "$M3_CONFIG" ]] || { echo "Expected absolute M3_CONFIG" >&2; exit 2; }
    ssh -S "$socket" -O check "$peer"
    timeout 2 bash -c 'true >"/dev/tcp/$1/5557"' _ "$host"
    if ss -H -ltn '( sport = :18642 or sport = :18643 )' | grep -q .; then echo "Model ports already occupied" >&2; exit 1; fi
    existing=$(ros2 node list --no-daemon --spin-time 1)
    if [[ -n "$existing" ]]; then echo "Use an empty isolated domain: $existing" >&2; exit 1; fi
    printf '%s\n' "$M3_CONFIG" "$M3_TARGET_ID" >"$run/profile"
    sha256sum "$M3_CONFIG" >"$run/config.sha256"
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv >"$run/gpu-before.txt"
    trap 'cleanup' ERR
    launch bridge python -u -c 'from alohamini_bridge.bridge_node import main; main()' --ros-args -p host:="$host" -p state_timestamp_mode:=request_bound -p command_timeout_sec:=0.20
    disable
    launch camera python -u examples/navigation/m3_camera_timing.py "$socket" "$peer" --ros-args -p host:="$host" -p 'camera_names:=[forward,chest]' -p publish_raw:=false -p timestamp_mode:=host_wall
    export M2B_LOG_DIR="$run"
    : >"$run/lightnav.log"
    : >"$run/depth.log"
    launch services bash examples/navigation/m2b_model_services.sh
    ready=false
    for ((i=0;i<120;i++)); do
        if ! owned services; then break; fi
        if grep -q 'READY host=' "$run/lightnav.log" 2>/dev/null && grep -q 'Metric3D ready' "$run/depth.log" 2>/dev/null; then ready=true; break; fi
        sleep 1
    done
    if ! "$ready"; then cleanup; echo "Model startup failed; inspect $run" >&2; exit 1; fi
    launch grid python -u examples/navigation/m2b_rgb_nodes.py grid --config "$M3_CONFIG"
    launch arrival python -u examples/navigation/m2b_rgb_nodes.py arrival --config "$M3_CONFIG" --color "$color" --target-id "$M3_TARGET_ID"
    launch controller python -u -m alohamini.cerebellum.ros_adapter --ros-args -p dry_run:=false -p enable_actuators:=true -p grid_source:=measured_metric -p config:="$M3_CONFIG" -p publish_capture_pose:=true -p target_id:="$M3_TARGET_ID" -p stop_confirm_count:=2
    launch brain python -u examples/navigation/m2b_rgb_nodes.py brain --config "$M3_CONFIG" --instruction "$instruction"
    sleep 2
    for role in camera bridge grid arrival controller brain services; do
        if ! owned "$role"; then cleanup; echo "$role exited during startup; check $run/$role.log" >&2; exit 1; fi
    done
    echo "Started with bridge commands disabled. Logs: $run"
    ;;
preflight|drive)
    [[ $# == 2 ]] || { echo "Supply a new output directory" >&2; exit 2; }
    exec 8>"$run/trial.lock"; flock -n 8 || exit 1
    for role in bridge services controller grid arrival camera brain; do owned "$role" || { echo "Missing $role" >&2; exit 1; }; done
    sha256sum -c "$run/config.sha256"
    [[ "$(cat "$run/profile")" == "$(printf '%s\n' "$M3_CONFIG" "$M3_TARGET_ID")" ]] || exit 2
    args=(--chair)
    if [[ "$action" == drive ]]; then
        printf 'Right detour about 0.6–0.9 m, then forward about 1.2–1.5 m. Clear the full route, physical E-stop in reach. Type GO: '
        read -r answer; [[ "$answer" == GO ]] || exit 2
        args+=(--operator-confirmed)
        trap 'disable' EXIT
    fi
    python -u examples/navigation/m3_trial_monitor.py "$2" "${args[@]}"
    ;;
estop)
    timeout 5 ros2 topic pub --once "$ctrl/estop" std_msgs/msg/Bool '{data: true}' || true
    disable
    ;;
stop)
    if owned bridge; then disable || { echo 'Disable not acknowledged; use physical E-stop and inspect the bridge.' >&2; exit 1; }; fi
    cleanup
    ;;
status)
    ros2 node list
    ros2 topic info -v /cmd_vel || true
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
    ;;
*) echo "Usage: bash $0 {start|preflight DIR|drive DIR|estop|stop|status}" >&2; exit 2 ;;
esac

