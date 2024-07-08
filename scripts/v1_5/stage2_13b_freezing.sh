#!/bin/bash
#SBATCH --job-name=llava_pret13b_stage2-lora_l128_xformer_residual_zero3 # Job name
#SBATCH --output=STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption.txt
#SBATCH --nodes=2
#SBATCH --nodelist=gpumid-21,gpumid-22
#SBATCH --mem=240G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/llava_finetune_data/hostfile_3.txt llava/train/train_mem.py \
    --deepspeed /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/llava_q_test/scripts_design/zero3.json \
    --model_name_or_path LLM360/CrystalChat \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Backup/llava_data/visionteam_lcdata_no_mimic_rdshuffled_0201.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/llava_data/pretrain_data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-7b-crystal_chat-short-caption/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
