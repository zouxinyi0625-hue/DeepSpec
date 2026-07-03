# MAI Profile DSpark Training Wrapper

Use `scripts/train_maiprofile/train_dspark_short_sync.sh` to train the first short-layer MAI Profile DSpark model.

The wrapper intentionally writes checkpoints locally first, then mirrors them to the MSN.DnI mount:

- Local checkpoint: `/home/aiscuser/checkpoints/deepspec_small/<EXP_NAME>`
- Remote checkpoint: `$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/checkpoints/deepspec_small/<EXP_NAME>`
- Local tensorboard: `/home/aiscuser/tensorboard/deepspec_small/<EXP_NAME>`
- Remote tensorboard: `$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/tensorboard/deepspec_small/<EXP_NAME>`

This avoids slow FSDP checkpoint writes directly to ADLS/Cosmos while still preserving artifacts if the VM/job disappears.

## Run

```bash
cd /scratch/azureml/cr/j/62762bfeddfd4c1b8e0df81ac7b09742/exe/wd/DeepSpec

git fetch
git checkout dev/maiprofile
git pull

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_TRAIN_STEPS=3000 \
CHECKPOINTING_STEPS=250 \
SYNC_INTERVAL_SECS=300 \
bash scripts/train_maiprofile/train_dspark_short_sync.sh
```

## Key environment variables

| variable | default | meaning |
|---|---|---|
| `EXP_NAME` | `dspark_block5_gemma4_12b_maiprofile_short_1024ctx_10k` | checkpoint/tensorboard experiment name |
| `TARGET_CACHE_DIR` | `$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/target_cache/20260615/gemma4_12b_maiprofile_short_layers` | target cache path |
| `MAX_TRAIN_STEPS` | `3000` | train steps |
| `CHECKPOINTING_STEPS` | `250` | save frequency |
| `SYNC_INTERVAL_SECS` | `300` | background sync interval |
| `RUN_SYNC_LOOP` | `1` | set `0` to disable periodic sync and only final sync |
| `SYNC_DELETE` | `0` | set `1` to make remote mirror exactly match local |

## Check sync status

```bash
EXP_NAME=dspark_block5_gemma4_12b_maiprofile_short_1024ctx_10k
BASE=${AZURE_ML_INPUT_msndni}/shares/users/zxy/maiprofile

tail -f ${BASE}/logs/${EXP_NAME}_sync.log
ls -lh ${BASE}/checkpoints/deepspec_small/${EXP_NAME}
```

## Final manual sync

The wrapper runs a final sync on normal exit or Ctrl-C. If needed, manually sync:

```bash
EXP_NAME=dspark_block5_gemma4_12b_maiprofile_short_1024ctx_10k
SRC=/home/aiscuser/checkpoints/deepspec_small/${EXP_NAME}
DST=${AZURE_ML_INPUT_msndni}/shares/users/zxy/maiprofile/checkpoints/deepspec_small/${EXP_NAME}
mkdir -p "$DST"
rsync -a "$SRC"/ "$DST"/
```
