#!/usr/bin/env bash
set -euo pipefail
root=/mnt/hdd2/yanbocheng/photo-goal-native
source "$root/environment.sh"
cd "$root"
if [[ ${1:-} == status ]]; then
    if [[ -f runs/overnight.pid ]]; then
        pid=$(cat runs/overnight.pid)
        ps -p "$pid" -o pid,etimes,args || true
    fi
    tail -n 8 runs/overnight-visual-bootstrap.log
    [[ ! -f data/visual-bootstrap/training-receipt.json ]] || cat data/visual-bootstrap/training-receipt.json
    exit 0
fi
if [[ -f runs/overnight.pid ]] && kill -0 "$(cat runs/overnight.pid)" 2>/dev/null; then
    printf 'An existing study window is still running.\n' >&2
    exit 1
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
snapshot="$root/runs/source-$stamp"
mkdir -p "$snapshot"
cp -a -- "$root/code/photo_goal" "$snapshot/photo_goal"
export PYTHONPATH="$snapshot"
if [[ -f runs/overnight-visual-bootstrap.log ]]; then
    cp -- runs/overnight-visual-bootstrap.log "runs/overnight-before-$stamp.log"
fi
nohup timeout --signal=TERM --kill-after=120s 8h "$root/env/bin/python" -u -m photo_goal \
    --root "$root" bootstrap-city --flights 128 --updates 20000 --hours 7.9 \
    >> runs/overnight-visual-bootstrap.log 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > runs/overnight.pid
printf 'Started resumable visual bootstrap window: PID %s\n' "$pid"
