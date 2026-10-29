#!/usr/bin/env bash
# Download PASTIS (Zenodo record 5012942: one file PASTIS.zip, 28.8 GB,
# md5 cfc441bf18137ff0bbf4fad58828fb98) straight to the UCloud drive, unzip it, check it,
# and write splits.json. Run on a CPU machine: no GPU needed. Restartable.
set -euo pipefail
source "$(dirname "$0")/config.sh"
ZIP="$DATA_DIR/download/PASTIS.zip"
MD5=cfc441bf18137ff0bbf4fad58828fb98

if [ ! -f "$PASTIS/metadata.geojson" ]; then
  mkdir -p "$DATA_DIR/download"
  if [ ! -f "$ZIP" ] || [ "$(md5sum "$ZIP" | cut -d' ' -f1)" != "$MD5" ]; then
    # zenodo_get checks the md5 itself; curl is the fallback (-C - resumes a broken download)
    zenodo_get 5012942 -o "$DATA_DIR/download" || \
      curl -L -C - -o "$ZIP" "https://zenodo.org/records/5012942/files/PASTIS.zip?download=1"
  fi
  echo "checking md5 (takes a minute)..."
  [ "$(md5sum "$ZIP" | cut -d' ' -f1)" = "$MD5" ] || { echo "md5 does not match: download again"; exit 1; }
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -n "$ZIP" -d "$DATA_DIR"
  else
    python -m zipfile -e "$ZIP" "$DATA_DIR"
  fi
fi

if [ ! -f "$PASTIS/metadata.geojson" ]; then
  echo "metadata.geojson not found in $PASTIS. Look in $DATA_DIR and run: export PASTIS=<folder>"
  exit 1
fi
python scripts/check_data.py --pastis "$PASTIS"
python scripts/make_splits.py --pastis "$PASTIS" --out "$SPLITS"
echo "PASTIS is ready. You can now delete $ZIP to save 29 GB."
echo "Share $SPLITS with Pair 1 so both pairs use the same label subsets."
