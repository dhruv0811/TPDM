#!/bin/bash
# Joint LoRA and TPDM training script
# This script trains both LoRA adapters for the UNet and the Time Prediction model

export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OMP_NUM_THREADS=4
export WANDB_PROJECT="tpdm_lora"
export WANDB_MODE="online"
export RUN_NAME="lora_tpdm_joint_training"

# Create output directory with timestamp
OUTPUT_DIR="outputs/$(date +'%Y-%m-%d')/$RUN_NAME"

# Choose model type (sd3 or sd1.5)
MODEL_TYPE="sd3"  # Change to "sd1.5" for Stable Diffusion 1.5

# Model configurations based on model type
if [ "$MODEL_TYPE" == "sd3" ]; then
  MODEL_CONFIG="configs/models/sd3_pnt_lora.yaml"
  INIT_ALPHA=2.5
  INIT_BETA=1.0
else
  MODEL_CONFIG="configs/models/sd1_5_pnt_lora.yaml"
  INIT_ALPHA=1.5
  INIT_BETA=-0.7
fi

# Launch distributed training
python -m torch.distributed.run --nproc_per_node $NUM_GPUS --nnodes 1 --standalone \
    main_lora_tpdm_trainer.py \
    --model_config $MODEL_CONFIG \
    --reward_model_config configs/models/image_reward.yaml \
    --train_dataset configs/datasets/hf_json_list.yaml \
    --data_collator configs/datasets/json_prompt_collator.yaml \
    --gamma 0.97 \
    --world_size $NUM_GPUS \
    --init_alpha $INIT_ALPHA \
    --init_beta $INIT_BETA \
    --kl_coef 0.00 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lora_learning_rate 1e-4 \
    --lora_weight_decay 0.01 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 100 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --adam_epsilon 1e-5 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1 \
    --eval_steps 50 \
    --save_steps 100 \
    --torch_empty_cache_steps 10 \
    --logging_steps 10 \
    --report_to wandb \
    --resume_from_checkpoint true \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --deepspeed configs/deepspeed/deepspeed_stage_0.json

echo '--------------------------'
echo 'Joint LoRA+TPDM training complete'
echo "LoRA weights saved to: $OUTPUT_DIR/lora_weights.pt"
echo '--------------------------'