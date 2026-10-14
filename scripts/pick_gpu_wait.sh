#!/bin/bash
# Wait (inside 1 MiB) for a GPU with enough free memory, then print its index.
#
#   PICK=$(bash scripts/pick_gpu_wait.sh ${MIN_FREE_MB:-20000} ${WAIT_MIN:-45}) || exit 0
#
# Unlike pick_gpu.sh (refuse-and-retry-later), this holds the PBS slot and polls
# until a GPU frees, so a queued job starts *by itself* the moment capacity
# appears - no external scheduler has to fire.  If the wait window expires the
# script exits non-zero and the launcher exits cleanly (resubmit later).
set -u
MIN_FREE_MB="${1:-20000}"
WAIT_MIN="${2:-45}"
DEADLINE=$((SECONDS + WAIT_MIN * 60))

while :; do
    if PICK=$(bash scripts/pick_gpu.sh "$MIN_FREE_MB"); then
        echo "pick_gpu_wait: GPU $PICK free >= ${MIN_FREE_MB} MiB (waited $((SECONDS / 60)) min)" >&2
        echo "$PICK"
        exit 0
    fi
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "pick_gpu_wait: no GPU with >= ${MIN_FREE_MB} MiB free within ${WAIT_MIN} min - exiting; resubmit later" >&2
        exit 3
    fi
    sleep 60
done
