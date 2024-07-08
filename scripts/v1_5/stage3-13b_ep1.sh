#!/bin/bash
#SBATCH --job-name=stage3_ep1 # Job name
#SBATCH --output=logs/stage3/llava_pret13b_stage3-fixed_ep1.txt
#SBATCH --nodes=4
#SBATCH --nodelist=gpumid-[06-07],gpumid-[12,13]
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

deepspeed --hostfile scripts_sm/hostfiles/4nodes_v4.txt llava/train/train_mem.py \
    --deepspeed /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/llava_q_test/scripts_design/zero3.json \
    --model_name_or_path /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/checkpoints/llava_pret13b_stage2-fixed_ep1 \
    --version v1 \
    --data_path /lustre/scratch/shared-folders/vision-project/Code/jinhong.wang/llava1.5_llama2/data/llava_finetune_data/llava_v1_5_mix665k.json \
    --image_folder /lustre/scratch/shared-folders/vision-project/Code/jinhong.wang/llava1.5_llama2/data/llava_finetune_data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/checkpoints/llava_pret13b_stage3-fixed_ep1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
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