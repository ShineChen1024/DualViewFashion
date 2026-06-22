export MODEL_NAME="/home/work/MMSearch/cwf/search/model/FLUX.1-Fill-dev"
export OUTPUT_DIR="/home/work/MMSearch/svg-shared-model-new-copy/ai-search/lora/trained-lora-multiview-caption-stage1-r1024"
export WANDB_PROJECT=finetune-multiview
export WANDB_NAME=run_multiview_caption_stage1

WORLD_SIZE=${WORLD_SIZE:-1}
echo "WORLD_SIZE=$WORLD_SIZE"
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12345}

ACC_STEPS=1

if [ "$WORLD_SIZE" -gt 1 ]; then
  MULTI_GPU_ARGS="--multi_gpu --num_processes=$WORLD_SIZE --num_machines=$WORLD_SIZE --machine_rank=$RANK --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT"
else
  MULTI_GPU_ARGS=""
fi

accelerate launch \
  $MULTI_GPU_ARGS \
  --mixed_precision=bf16 finetune-multiview-caption.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --cloth_size=512 \
  --rank=1024 \
  --lora_alpha=2048 \
  --train_batch_size=4 \
  --num_train_epochs=20 \
  --guidance_scale=1. \
  --gradient_accumulation_steps=$ACC_STEPS \
  --gradient_checkpointing \
  --optimizer="AdamW" \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --checkpointing_steps=2000 \
  --seed="42"
