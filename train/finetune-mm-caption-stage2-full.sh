export MODEL_NAME="/home/work/MMSearch/cwf/search/model/FLUX.1-Fill-dev"
export STAGE1_MODEL_DIR="/home/work/MMSearch/svg-shared-model-new-copy/ai-search/full/trained-full-multiview-caption-stage1"
export STAGE1_TRANSFORMER_DIR="${STAGE1_MODEL_DIR}/transformer"
export OUTPUT_DIR="/home/work/MMSearch/svg-shared-model-new-copy/ai-search/full/trained-full-multiview-caption-stage2"
export WANDB_PROJECT=finetune-multiview
export WANDB_NAME=run_multiview_caption_stage2_full

WORLD_SIZE=${WORLD_SIZE:-1}
echo "WORLD_SIZE=$WORLD_SIZE"
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29502}

ACC_STEPS=1

if [ ! -d "$STAGE1_TRANSFORMER_DIR" ]; then
  echo "Stage1 transformer not found: $STAGE1_TRANSFORMER_DIR" >&2
  echo "Run finetune-mm-caption-stage1-full.sh to completion before starting stage2." >&2
  exit 1
fi

if [ "$WORLD_SIZE" -gt 1 ]; then
  LAUNCH_CMD="torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
  DS_ARGS="--deepspeed_config=ds_config_zero2.json"
else
  LAUNCH_CMD="accelerate launch --mixed_precision=bf16"
  DS_ARGS=""
fi

LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"

$LAUNCH_CMD finetune-multiview-caption-full-deepspeed.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_model_name_or_path=$STAGE1_TRANSFORMER_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --cloth_size=1024 \
  --train_batch_size=1 \
  --num_train_epochs=10 \
  --guidance_scale=1. \
  --gradient_accumulation_steps=$ACC_STEPS \
  --gradient_checkpointing \
  --optimizer="AdamW" \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=200 \
  --checkpointing_steps=1000 \
  --seed="42" \
  $DS_ARGS \
  2>&1 | tee "$LOG_FILE"
