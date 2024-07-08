#!/bin/bash
#SBATCH --job-name=pretrain_crystal_chat_llava # Job name
#SBATCH --output=stage1_and_2_combined_pretrain_Short_caption_crystal_chat_llava.txt
#SBATCH --nodes=4
#SBATCH --nodelist=gpumid-21,gpumid-22,gpumid-23,gpumid-24
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile_new.txt llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path LLM360/CrystalChat \
    --version plain \
    --data_path /lustre/scratch/shared-folders/vision-project/Backup/llava_data/visionteam_1_2_rdshuffled_0222.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/data/image  \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type ours \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-stage1_and_2_combined_pretrain_crystal_chat-short-caption \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
