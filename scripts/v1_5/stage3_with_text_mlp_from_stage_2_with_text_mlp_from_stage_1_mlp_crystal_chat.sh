#!/bin/bash
#SBATCH --job-name=stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat_1epoch # Job name
#SBATCH --output=stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat_1epoch.txt
#SBATCH --nodes=4
#SBATCH --nodelist=gpumid-21,gpumid-22,gpumid-23,gpumid-24
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile.txt llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/stage_2_with_text_old_projector_from_pretrain_mlp_stage_1_data \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Backup/data/text_add2/stage3_w_text.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/data/image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat_1epoch \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
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
    --report_to wandb
