#!/bin/bash
#SBATCH --job-name=mmeeval # Job name
#SBATCH --output=mme-scalingins13b.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path ckpt_scaling/llava_scaling1_instune13b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava_scaling1_instune13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava_scaling1_instune13b
cd eval_tool

python calculation.py --results_dir answers/llava_scaling1_instune13b