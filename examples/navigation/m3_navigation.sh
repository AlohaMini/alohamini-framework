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
if [[ -z "${M3_NATIVE_REPO:-}" && ! -d "$native/ros2/src/alohamini_camera" ]]; then
    native="$jjy/code/Alohamini_software-lightnav"
fi
export PYTHONPATH="$repo/src:$repo/ros2/src/alohamini_bridge:$native/ros2/src/alohamini_camera:$repo/examples/navigation"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
requested_config=${M3_CONFIG:-}
export M3_CONFIG=${M3_CONFIG:-"$repo/evidence/m3-low-speed-shared.json"}
host=${M3_PI_HOST:-192.168.2.2}
peer=${M3_PI_USER:-pi5}@$host
socket=${M3_SSH_SOCKET:-"/tmp/jjy-m3-$UID-$ROS_DOMAIN_ID-$host"}
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
    registering=true
    nohup setsid "$@" </dev/null >"$run/$role.log" 2>&1 8>&- &
    local pid=$!
    echo "$pid $(awk '{print $22}' /proc/$pid/stat)" >"$run/$role.pid"
    registering=false
    if ((startup_signal)); then exit "$startup_signal"; fi
}
startup_interrupted() {
    startup_signal=$1
    if ! "$registering"; then exit "$startup_signal"; fi
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
    local role pid n stat birth entry
    local -a sessions=()
    for role in brain controller arrival grid services camera bridge; do
        if owned "$role"; then
            read -r pid birth <"$run/$role.pid"
            sessions+=("$pid:$birth")
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for entry in "${sessions[@]}"; do
        pid=${entry%:*}
        birth=${entry#*:}
        # A model child may outlive the supervisor in its setsid group.
        for ((n=0; n<70; n++)); do
            if [[ -r "/proc/$pid/stat" && "$(awk '{print $22}' "/proc/$pid/stat")" != "$birth" ]]; then break; fi
            stat=$(ps -o stat= -s "$pid" 2>/dev/null) || break
            if ! grep -q '^[[:space:]]*[^Z[:space:]]' <<< "$stat"; then break; fi
            if ((n==60)); then kill -KILL -- "-$pid" 2>/dev/null || true; fi
            sleep .1
        done
        if ((n==70)); then echo "Live processes remain in session $pid" >&2; return 1; fi
        wait "$pid" 2>/dev/null || true
    done
    for role in brain controller arrival grid services camera bridge; do
        if owned "$role"; then echo "Shutdown unconfirmed: $role" >&2; return 1; fi
        rm -f -- "$run/$role.pid"
    done
}
action=${1:-status}
session=false
if [[ "$action" == run ]]; then
    [[ $# == 2 && -n "$2" ]] || { echo "Usage: bash $0 run 'navigation instruction'" >&2; exit 2; }
    export M3_INSTRUCTION="$2"
    session=true
    action=start
    set -- start
fi
chair=false
if [[ "$action" == start || "$action" == restart || "$action" == preflight || "$action" == drive ]]; then
    offset=2
    if [[ "$action" == preflight || "$action" == drive ]]; then
        [[ -n "${2:-}" && "$2" != --* ]] || { echo "Supply a new output directory" >&2; exit 2; }
        offset=3
    fi
    case "${*:$offset}" in
        '') ;;
        --chair) chair=true ;;
        *) echo "Only --chair is supported after the positional arguments." >&2; exit 2 ;;
    esac
fi
case "$action" in
start|restart)
    dry=${M3_DRY_RUN:-false}
    case "$dry" in false) actuate=true;; true) actuate=false;; *) echo "Invalid dry-run mode" >&2; exit 2;; esac
    color=${M3_COLOR:-}
    if "$chair"; then color=${color:-purple}; fi
    case "$color" in ''|purple|blue|green) ;; *) echo "Unsupported M3_COLOR: $color" >&2; exit 2 ;; esac
    export M3_TARGET_ID=${M3_TARGET_ID:-}
    instruction=${M3_INSTRUCTION:-}
    target_args=()
    roles=(camera bridge grid controller brain services)
    if [[ -n "$color" ]]; then
        export M3_COLOR="$color"
        export M3_TARGET_ID=${M3_TARGET_ID:-"${color}_toy"}
        target_args=(-p "target_id:='$M3_TARGET_ID'")
        roles+=(arrival)
        instruction=${instruction:-"Approach the $color toy and stop in front of it."}
    elif [[ -n "$M3_TARGET_ID" ]]; then
        echo "Set M3_COLOR when selecting a metric target." >&2; exit 2
    else
        unset M3_COLOR
        instruction=${instruction:-"Move forward through the open space."}
    fi
    exec 8>"$run/start.lock"; flock -n 8 || exit 1
    printf '%s %s\n' "$$" "$(awk '{print $22}' /proc/$$/stat)" >"$run/startup.pid"
    trap 'rm -f -- "$run/startup.pid"' EXIT
    if [[ "$action" == restart ]]; then
        # A supervisor holding the trial lock may still be disabling its bridge.
        ( flock -n 9 || { echo "Active trial: stop it before restarting." >&2; exit 1; }
          if owned bridge; then disable; fi
          cleanup
        ) 9>"$run/trial.lock"
    fi
    for role in bridge services controller camera grid arrival brain; do
        if owned "$role"; then echo "$role is already running in $run" >&2; exit 1; fi
        rm -f -- "$run/$role.pid"
    done
    [[ "$M3_CONFIG" = /* && -f "$M3_CONFIG" ]] || { echo "Expected absolute M3_CONFIG" >&2; exit 2; }
    if ! ssh -S "$socket" -O check "$peer" 2>/dev/null; then
        if [[ "$host" == 192.168.2.2 && -z "${M3_SSH_SOCKET:-}" ]] && ssh -S /tmp/jjy-m34-hw -O check "$peer" 2>/dev/null; then
            socket=/tmp/jjy-m34-hw
        else
            ssh -MNf -S "$socket" -o ConnectTimeout=5 -o ControlPersist=2h -o StrictHostKeyChecking=yes "$peer"
        fi
    fi
    ssh -S "$socket" -O check "$peer"
    timeout 2 bash -c 'true >"/dev/tcp/$1/5557"' _ "$host" || { echo "Pi Host unavailable at $host:5557" >&2; exit 1; }
    if ss -H -ltn '( sport = :18642 or sport = :18643 )' | grep -q .; then echo "Model ports already occupied" >&2; exit 1; fi
    existing=$(ros2 node list --no-daemon --spin-time 1)
    if [[ -n "$existing" ]]; then echo "Use an empty isolated domain: $existing" >&2; exit 1; fi
    printf '%s\n' "$M3_CONFIG" "$M3_TARGET_ID" "$color" >"$run/profile"
    printf '%s' "$instruction" >"$run/instruction"
    printf '%s\n' "$dry" >"$run/dry-run"
    printf '%s\n' "${M3_SESSION_TOKEN:-standalone}" >"$run/session"
    sha256sum "$M3_CONFIG" >"$run/config.sha256"
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv >"$run/gpu-before.txt"
    # Install rollback before launching detached processes. ERR alone misses Ctrl-C.
    registering=false
    startup_signal=0
    trap 'cleanup; rm -f -- "$run/startup.pid"' EXIT
    trap 'startup_interrupted 130' INT
    trap 'startup_interrupted 143' TERM
    trap 'startup_interrupted 129' HUP
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
    if ! "$ready"; then echo "Model startup failed; inspect $run" >&2; exit 1; fi
    launch grid python -u examples/navigation/m2b_rgb_nodes.py grid --config "$M3_CONFIG"
    if [[ -n "$color" ]]; then
        launch arrival python -u examples/navigation/m2b_rgb_nodes.py arrival --config "$M3_CONFIG" --color "$color" --target-id "$M3_TARGET_ID"
    fi
    launch controller python -u -m alohamini.cerebellum.ros_adapter --ros-args -p dry_run:="$dry" -p enable_actuators:="$actuate" -p grid_source:=measured_metric -p config:="$M3_CONFIG" -p publish_capture_pose:=true "${target_args[@]}" -p stop_confirm_count:=2
    launch brain python -u examples/navigation/m2b_rgb_nodes.py brain --config "$M3_CONFIG" --instruction "$instruction"
    sleep 2
    for role in "${roles[@]}"; do
        if ! owned "$role"; then echo "$role exited during startup; check $run/$role.log" >&2; exit 1; fi
    done
    echo "Started with bridge commands disabled. Logs: $run"
    rm -f -- "$run/startup.pid"
    flock -u 8
    if "$session"; then
        # drive handles estop and disable; check the bridge acknowledgement again before teardown.
        trap 'if disable; then cleanup; else echo "Bridge disable unconfirmed. Use physical E-stop and inspect bridge." >&2; exit 1; fi' EXIT
        bash "$repo/examples/navigation/m3_navigation.sh" drive "trials/free-$(date +%Y%m%d-%H%M%S)-$$"
    else
        trap - EXIT
    fi
    ;;
preflight|drive)
    exec 8>"$run/trial.lock"; flock -n 8 || exit 1
    mapfile -t profile <"$run/profile"
    if [[ -z "$requested_config" && ${#profile[@]} == 3 ]]; then export M3_CONFIG="${profile[0]}"; fi
    [[ ${#profile[@]} == 3 && "${profile[0]}" == "$M3_CONFIG" ]] || { echo "Restart services with the current profile." >&2; exit 2; }
    if [[ ${M3_COLOR-${profile[2]}} != "${profile[2]}" || ${M3_TARGET_ID-${profile[1]}} != "${profile[1]}" ]]; then
        echo "Metric target changed; stop and start services with the new target." >&2; exit 2
    fi
    export M3_TARGET_ID="${profile[1]}"
    instruction=$(cat "$run/instruction")
    if [[ -n "${M3_INSTRUCTION:-}" && "$M3_INSTRUCTION" != "$instruction" ]]; then
        echo "Instruction changed; stop and start services with the new instruction." >&2; exit 2
    fi
    roles=(bridge services controller grid camera brain)
    if [[ -n "$M3_TARGET_ID" ]]; then roles+=(arrival); fi
    for role in "${roles[@]}"; do owned "$role" || { echo "Missing $role" >&2; exit 1; }; done
    sha256sum -c "$run/config.sha256"
    args=()
    dry=$(cat "$run/dry-run")
    [[ "$dry" == true || "$dry" == false ]] || exit 2
    [[ "${M3_DRY_RUN:-$dry}" == "$dry" ]] || { echo "Dry-run mode differs from the startup profile." >&2; exit 2; }
    if [[ "$dry" == true ]]; then args+=(--observe-only-graph); fi
    if "$chair"; then
        [[ -n "$M3_TARGET_ID" ]] || { echo "Chair acceptance requires start --chair or M3_COLOR at startup." >&2; exit 2; }
        args+=(--chair)
    fi
    if [[ "$action" == drive && "$dry" == true ]]; then
        args+=(--dry-run)
        echo "Observing only. Bridge commands disabled; Ctrl-C ends recording."
    elif [[ "$action" == drive ]]; then
        if "$chair"; then echo 'Right detour about 0.6–0.9 m, then forward about 1.2–1.5 m.'
        else printf 'Free navigation: %s\n' "$instruction"; fi
        printf 'Clear the route, physical E-stop in reach. Type GO: '
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
    exec 8>"$run/start.lock"
    if [[ -n "${M3_STOP_TOKEN:-}" ]]; then
        flock -w 15 8 || exit 1
        [[ -r "$run/session" && "$(cat "$run/session")" == "$M3_STOP_TOKEN" ]] || exit 0
    elif ! flock -n 8; then
        # Stop a partially launched transaction before it can create more roles.
        if owned startup; then
            read -r pid _ <"$run/startup.pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
        flock -w 20 8 || { echo "Startup still owns the lock; inspect startup.pid." >&2; exit 1; }
    fi
    rm -f -- "$run/startup.pid"
    if owned bridge; then disable || { echo 'Disable not acknowledged; use physical E-stop and inspect the bridge.' >&2; exit 1; }; fi
    cleanup
    ;;
status)
    ros2 node list
    ros2 topic info -v /cmd_vel || true
    nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
    ;;
*) echo "Usage: bash $0 {run 'instruction'|start [--chair]|restart [--chair]|preflight DIR [--chair]|drive DIR [--chair]|estop|stop|status}" >&2; exit 2 ;;
esac

