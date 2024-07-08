#!/bin/bash 
#SBATCH --job-name=a_mmellava_scaling2_instune13b_checkpoint-10394
#SBATCH --output=eval_mme_llava_scaling2_instune13b_checkpoint-10394.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path ckpt_scaling/llava_scaling2_instune13b/checkpoint-10394 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava_scaling2_instune13b_checkpoint-1039412.22-17.23.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava_scaling2_instune13b_checkpoint-1039412.22-17.23

cd eval_tool 

python calculation.py --results_dir answers/llava_scaling2_instune13b_checkpoint-1039412.22-17.23