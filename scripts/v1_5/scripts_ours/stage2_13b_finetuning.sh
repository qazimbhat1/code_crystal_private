#!/bin/bash
#SBATCH --job-name=llava_pret13b_stage2-lora_l128_xformer_residual_zero3 # Job name
#SBATCH --output=logs/stage2/llava_pret13b_stage2-lora_l128_xformer_residual_zero3.txt
#SBATCH --nodes=4
#SBATCH --nodelist=gpumid-[05-08]
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile scripts_sm/hostfiles/4nodes_v1.txt llava/train/train_mem.py \
    --deepspeed /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/llava_q_test/scripts_design/zero3.json \
    --model_name_or_path /lustre/scratch/shared-folders/vision-project/Code/jinhong.wang/llava1.5_llama2/pretweight/llama2_13b_hfweight \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Backup/llava_data/finetune_json/MIMIC-IT/submimic_llava.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Backup/llava_data/llava_finetune_image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type xformer_residual \
    --pretrain_mm_mlp_adapter /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/checkpoints/llava_pret13b_pretrain_l128_xformer_residual/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/checkpoints/llava_pret13b_stage2-lora_l128_xformer_residual_zero3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
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
