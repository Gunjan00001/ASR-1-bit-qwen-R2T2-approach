#!/bin/bash
# Shared GPU picker for the JIIT A100 cluster (run on a compute node - the
# login node has no driver).
#
#   PICK=$(bash scripts/pick_gpu.sh ${MIN_FREE_MB:-20000}) || exit 0
#   export CUDA_VISIBLE_DEVICES="$PICK"
#
# Samples the GPU state three times, picks the GPU with the most free memory,
# and *refuses* (exit 3, empty stdout) when nothing has MIN_FREE_MB free - the
# cluster's co-tenants grow mid-run, so a "least-used" fallback only buys a
# doomed job (2026-10-15: two aligned runs OOM'd at their first step after half
# an hour of forced-aligner work). A refused job exits cleanly; resubmit when
# the cluster frees instead of burning the slot.
set -u
MIN_FREE_MB="${1:-20000}"

OUT=$(for _ in 1 2 3; do
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null
    sleep 1
  done | awk -F', ' -v min="$MIN_FREE_MB" '
    NF >= 3 && $3 + 0 > 0 {
      idx = $1 + 0; used = $2 + 0; free = ($3 + 0) - used
      if (!(idx in best) || free > best[idx]) { best[idx] = free; used_at[idx] = used }
      if (!(seen) || free > maxfree) { maxfree = free; maxidx = idx; seen = 1 }
    }
    END {
      if (seen && maxfree >= min) print maxidx;
      else { printf "pick_gpu: best free %d MiB < %d MiB - resubmit later\n", maxfree + 0, min > "/dev/stderr"; exit 3 }
    }')

STATUS=$?
if [ $STATUS -ne 0 ] || [ -z "$OUT" ]; then
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null >&2
    exit 3
fi
echo "$OUT"
