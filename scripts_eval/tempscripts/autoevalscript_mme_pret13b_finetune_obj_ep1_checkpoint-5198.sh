#!/bin/bash 
#SBATCH --job-name=a_mmepret13b_finetune_obj_ep1_checkpoint-5198
#SBATCH --output=eval_mme_pret13b_finetune_obj_ep1_checkpoint-5198.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python /lustre/home/sukmin.yun/llavadevelop/llava/eval/model_vqa_loader.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/sukmin.yun/checkpoints/llava_pret13b_finetune_obj_ep1/checkpoint-5198 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/pret13b_finetune_obj_ep1_checkpoint-519812.26-12.48.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME 

python convert_answer_to_mme.py --experiment pret13b_finetune_obj_ep1_checkpoint-519812.26-12.48

cd eval_tool 

python calculation.py --results_dir answers/pret13b_finetune_obj_ep1_checkpoint-519812.26-12.48