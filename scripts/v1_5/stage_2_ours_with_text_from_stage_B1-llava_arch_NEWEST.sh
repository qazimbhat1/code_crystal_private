#!/bin/bash
#SBATCH --job-name=stage_2_ours_with_text_from_stage_B1-llava_arch_NEWEST # Job name
#SBATCH --output=stage_2_ours_with_text_from_stage_B1-llava_arch_NEWEST.txt
#SBATCH --nodes=4
#SBATCH --nodelist=gpumid-03,gpumid-04,gpumid-05,gpumid-06
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile_new.txt llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path LLM360/CrystalChat \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Backup/data/text_add2/stage2_w_text.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/data/image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type ours \
    --pretrain_mm_mlp_adapter /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-7b-stage1_B1_llava_arch_fixed_NEWEST/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/stage_2_ours_with_text_from_stage_B1-llava_arch_NEWEST \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
