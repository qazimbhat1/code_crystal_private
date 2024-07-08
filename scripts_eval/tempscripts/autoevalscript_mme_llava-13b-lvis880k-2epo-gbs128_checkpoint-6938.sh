#!/bin/bash 
#SBATCH --job-name=a_mmellava-13b-lvis880k-2epo-gbs128_checkpoint-6938
#SBATCH --output=eval_mme_llava-13b-lvis880k-2epo-gbs128_checkpoint-6938.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/lrvset/llava-13b-lvis880k-2epo-gbs128/checkpoint-6938 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-13b-lvis880k-2epo-gbs128_checkpoint-693812.26-14.01.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava-13b-lvis880k-2epo-gbs128_checkpoint-693812.26-14.01

cd eval_tool 

python calculation.py --results_dir answers/llava-13b-lvis880k-2epo-gbs128_checkpoint-693812.26-14.01