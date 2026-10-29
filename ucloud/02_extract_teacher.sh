#!/usr/bin/env bash
# Save DINOv3-SAT tokens for all PASTIS images (needed by BOTH pairs; do this first).
#   bash ucloud/02_extract_teacher.sh quick    # first 25 series (~1,000 images) so Pair 1 can start
#   bash ucloud/02_extract_teacher.sh          # all 2,433 series (restartable, skips finished ones)
#   bash ucloud/02_extract_teacher.sh dinov2   # open DINOv2 stand-in while DINOv3 access is pending
set -euo pipefail
source "$(dirname "$0")/config.sh"
MODE="${1:-full}"

case "$MODE" in
  quick)  python scripts/extract_teacher.py --pastis "$PASTIS" --out "$TEACHER_FEATS" \
            --model "$TEACHER_MODEL" --limit 25 ;;
  full)   python scripts/extract_teacher.py --pastis "$PASTIS" --out "$TEACHER_FEATS" \
            --model "$TEACHER_MODEL" ;;
  dinov2) python scripts/extract_teacher.py --pastis "$PASTIS" --out "$FEATS/dinov2_large" \
            --model facebook/dinov2-large --size 224 ;;
  *) echo "usage: $0 [quick|full|dinov2]"; exit 1 ;;
esac
du -sh "$FEATS"/* 2>/dev/null || true
