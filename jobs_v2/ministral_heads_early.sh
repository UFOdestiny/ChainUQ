#!/usr/bin/env bash
# Run Ministral's heads stage on GPU 2 as soon as its judging is done and GPU 2 is free,
# instead of waiting for GPU 3. The stage lock makes the later scheduled invocation a fast no-op.
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
T="$CUQ_ARTIFACTS/logs/full_ministral/timeline.txt"
until grep -q "ministral judge done" "$T" 2>/dev/null; do sleep 60; done
while ps -u "$USER" -o cmd | grep -q "[f]ull_backbone.sh ministral gen"; do sleep 60; done
TRAIN_GPU=$GEN_GPU jobs_v2/full_backbone.sh ministral heads
