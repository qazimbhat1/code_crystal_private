#!/bin/bash
#SBATCH --job-name=finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm # Job name
#SBATCH --output=finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm.txt
#SBATCH --nodes=4
#SBATCH --nodelist=gpumid-03,gpumid-04,gpumid-05,gpumid-06
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile_new.txt llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/Llama-2-7b-hf-new/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/ \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_final_v1_5_Unified_web_style_Junbo10k_pix2code_Junbo50k_QA_150k_+_LLAVA_Full.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/data/image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-llama2_7b_hfweight-pretrain-same_as_crystal_coder/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm \
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
