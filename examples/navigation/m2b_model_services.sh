#!/usr/bin/env bash
# Start the audited 5090 GPU services in the foreground supervisor's process group.
set -eu
: "${M2B_OPENVO:?path to the OpenVO checkout with Metric3D weights}"
: "${M2B_LIGHTNAV_MODEL:?path to the LightNav-0 checkpoint}"
: "${M2B_GPU_ENV:?path to the CUDA environment}"
: "${M2B_LOG_DIR:?path to the run logs}"
mkdir -p "$M2B_LOG_DIR"
here=$(cd "$(dirname "$0")" && pwd)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1
gpu="$M2B_GPU_ENV/bin"
"$gpu/python" -u "$here/m2b_depth_server.py" --openvo "$M2B_OPENVO" --width "${M2B_DEPTH_WIDTH:-384}" >"$M2B_LOG_DIR/depth.log" 2>&1 &
depth=$!
brain=""
cleanup() {
    if [[ -n "$brain" ]]; then kill "$brain" 2>/dev/null || true; fi
    kill "$depth" 2>/dev/null || true
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
"$gpu/python" -u -c 'import torch,runpy,sys;torch.cuda.set_per_process_memory_fraction(.39);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' "$gpu/lightnav-serve" --task vln --backend hf --num_history_frames 8 --model_path "$M2B_LIGHTNAV_MODEL" --host 127.0.0.1 --port 18643 >"$M2B_LOG_DIR/lightnav.log" 2>&1 &
brain=$!
wait -n "$depth" "$brain"
