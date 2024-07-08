#!/bin/bash
#SBATCH --job-name=pretrain_crystal_chat_llava # Job name
#SBATCH --output=latest____finetune_crystal_chat_old_data_+code+all_pretrain_stage1and2_combined.txt
#SBATCH --nodes=4
#SBATCH --nodelist=gpumid-21,gpumid-22,gpumid-23,gpumid-24
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile_new.txt llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path LLM360/CrystalChat \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_final_v1_5_WebSRC_+_Unified_web_style_WebSight_Junbo10k_pix2code_Junbo50k_QA_150k_+_LLAVA_Full.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/data/image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-7b-stage1_and_2_combined_pretrain_crystal_chat-short-caption/mm_projector.bin \
    --mm_projector_type ours \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-latest____finetune_crystal_chat_old_data_+code+all_pretrain_stage1and2_combined \
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
