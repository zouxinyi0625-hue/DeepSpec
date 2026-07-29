#!/usr/bin/env bash
set -euo pipefail

# Diagnostic wrapper: run the 26B trainer with faulthandler + NCCL/CUDA debug so
# that if it hangs, we get the exact Python stack of every rank instead of
# guessing from nvidia-smi.
#
# Usage:
#   bash scripts/train_maiprofile/debug_run_26b.sh
# When it hangs, either wait for the faulthandler timeout dump, or Ctrl-\ (SIGQUIT)
# to force a stack dump of all threads immediately.

# faulthandler: dump ALL thread stacks of every rank after N seconds of no
# progress, and on SIGQUIT/SIGABRT. This pinpoints the hung line.
export PYTHONFAULTHANDLER=1
# Blocking CUDA so any CUDA error surfaces at the real call site (not later).
export CUDA_LAUNCH_BLOCKING=1
# NCCL: verbose + async error handling so a stuck collective is reported.
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Shorten the collective watchdog so a hang aborts (with a stack) in ~3 min
# instead of the default 30-180 min.
export TORCH_NCCL_TIMEOUT_SEC=${TORCH_NCCL_TIMEOUT_SEC:-180}

# Force single-process dataloading (mount multi-worker deadlocks).
export NUM_WORKERS=0
export MASTER_PORT=${MASTER_PORT:-29544}

echo "Running 26B trainer in DEBUG mode (faulthandler + CUDA_LAUNCH_BLOCKING + NCCL blocking-wait)."
echo "If it hangs: press Ctrl-\\ (SIGQUIT) to dump every thread's Python stack,"
echo "or wait for the ${TORCH_NCCL_TIMEOUT_SEC}s NCCL watchdog to abort with a trace."

exec bash scripts/train_maiprofile/train_dspark_26b_sync.sh
