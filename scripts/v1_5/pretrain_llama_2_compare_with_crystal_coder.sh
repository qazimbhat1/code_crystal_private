#!/bin/bash
#SBATCH --job-name=pretrain_llama_2_compare_with_crystal_coder # Job name
#SBATCH --output=pretrain_llama_2_compare_with_crystal_coder.txt
#SBATCH --nodes=2
#SBATCH --nodelist=gpumid-07,gpumid-25
#SBATCH --mem=490G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --reservation=vision
# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile_3.txt llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/Llama-2-7b-hf-new/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/ \
    --version $PROMPT_VERSION \
    --data_path /lustre/scratch/shared-folders/vision-project/Backup/llava_data/pretrain_json/LLaVA_558k/blip_laion_cc_sbu_558k.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/llava_data/llava_pretrain_image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-llama2_7b_hfweight-pretrain-same_as_crystal_coder \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
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