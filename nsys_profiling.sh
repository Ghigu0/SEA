#!/usr/bin/env bash
set -e

NSYS=/opt/nvidia/nsight-systems/2026.1.1/bin/nsys



$NSYS profile \
  --force-overwrite true \
  --trace=cuda \
  --stats=true \
  -o sea_first \
  -- \
  ./sea ./database/ ./database/Video_3/frame_000130.raw --topk 50

