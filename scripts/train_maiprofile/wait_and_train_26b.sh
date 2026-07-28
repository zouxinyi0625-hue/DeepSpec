#!/usr/bin/env bash
set -euo pipefail

# Wait for the DSpark 26B target cache to finish, then auto-launch training.
#
# prepare_target_cache.py writes manifest.json ONLY after the finalize step
# succeeds on all ranks, so its existence is the reliable "cache is ready"
# signal. This script polls for it and, once present, kicks off the 26B trainer.
#
# Fire-and-forget before you leave: it blocks harmlessly until the cache lands,
# then runs training. Run it under nohup/tmux so it survives your logout:
#   nohup bash scripts/train_maiprofile/wait_and_train_26b.sh > wait_train.log 2>&1 &

# --- Resolve the data mount (handle both spellings/casings) -----------------
MOUNT="${MOUNT:-${AZURE_ML_INPUT_UKWDATA:-${AZURE_ML_INPUT_UKDATA:-${AZURE_ML_INPUT_ukwdata:-}}}}"
if [[ -z "${MOUNT}" ]]; then
    echo "ERROR: no data mount found. Set MOUNT=/path/to/ukwdata explicitly." >&2
    exit 1
fi

# Must match the cache-gen launcher's OUTPUT_DIR (same CACHE_TAG!).
CACHE_TAG=${CACHE_TAG:-v2}
TARGET_CACHE_DIR=${TARGET_CACHE_DIR:-${MOUNT}/maiprofile/dspark_26b/target_cache_${CACHE_TAG}}
MANIFEST="${TARGET_CACHE_DIR}/manifest.json"

POLL_SECS=${POLL_SECS:-120}
TIMEOUT_HOURS=${TIMEOUT_HOURS:-24}
max_polls=$(( TIMEOUT_HOURS * 3600 / POLL_SECS ))

echo "Waiting for 26B target cache to finish..."
echo "  cache dir: ${TARGET_CACHE_DIR}"
echo "  watching:  ${MANIFEST}"
echo "  poll:      every ${POLL_SECS}s, timeout ${TIMEOUT_HOURS}h"

polls=0
while [[ ! -f "${MANIFEST}" ]]; do
    polls=$(( polls + 1 ))
    if (( polls > max_polls )); then
        echo "ERROR: timed out after ${TIMEOUT_HOURS}h; manifest still missing: ${MANIFEST}" >&2
        exit 1
    fi
    # Light progress signal: shards written so far live under _tmp/rank_*/.
    n_shards=$(ls -1 "${TARGET_CACHE_DIR}"/_tmp/rank_*/*.bin 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date --iso-8601=seconds)] cache not ready (poll ${polls}/${max_polls}, shards so far=${n_shards}); sleeping ${POLL_SECS}s"
    sleep "${POLL_SECS}"
done

echo "[$(date --iso-8601=seconds)] manifest found — cache is ready. Sample count:"
python - "$MANIFEST" <<'PY' || true
import json, sys
m = json.load(open(sys.argv[1]))
print("  num_samples:", m.get("num_samples"))
print("  target_layer_ids:", m.get("target_layer_ids"))
print("  hidden_size:", m.get("hidden_size"))
PY

echo "Launching 26B training with TARGET_CACHE_DIR=${TARGET_CACHE_DIR}"
export TARGET_CACHE_DIR
exec bash scripts/train_maiprofile/train_dspark_26b_sync.sh
