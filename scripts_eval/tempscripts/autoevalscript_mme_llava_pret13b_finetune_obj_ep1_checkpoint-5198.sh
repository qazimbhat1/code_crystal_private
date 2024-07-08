#!/bin/bash 
#SBATCH --job-name=a_mmellava_pret13b_finetune_obj_ep1_checkpoint-5198
#SBATCH --output=eval_mme_llava_pret13b_finetune_obj_ep1_checkpoint-5198.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python /lustre/home/sukmin.yun/llavadevelop/llava/eval/model_vqa_loader.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/checkpoints/llava_pret13b_finetune_obj_ep1/checkpoint-5198 \
    --question-file /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/answers/llava_pret13b_finetune_obj_ep1_checkpoint-519812.26-13.44.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME 

python /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/convert_answer_to_mme.py --experiment /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/llava_pret13b_finetune_obj_ep1_checkpoint-519812.26-13.44

cd /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/eval_tool 

python /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/eval_tool/calculation.py --results_dir /lustre/home/sukmin.yun/llavadevelop/playground/data/eval/MME/eval_tool/answers/llava_pret13b_finetune_obj_ep1_checkpoint-519812.26-13.44