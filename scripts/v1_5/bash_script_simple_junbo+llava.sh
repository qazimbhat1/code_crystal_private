#!/bin/bash
#SBATCH --job-name=ft13b_rerun # Job name
#SBATCH --output=junbo-simple+llava.txt
#SBATCH --nodes=2
#SBATCH --nodelist=gpumid-15,gpumid-16
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile.txt llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/llava_final_v1_5_junbo_simple_+_llava.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-junbo-simple-+llava \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \

