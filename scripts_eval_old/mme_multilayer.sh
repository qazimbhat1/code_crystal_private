#!/bin/bash
#SBATCH --job-name=mmeeval # Job name
#SBATCH --output=mme-llava-13b-multilayer-output(-2-6).txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava-v1.5-llama2-13b-finetune-gbs128-multilayer-output \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-llama2-13b-finetune-gbs128-multilayer-output-2-6.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-llama2-13b-finetune-gbs128-multilayer-output-2-6
cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-llama2-13b-finetune-gbs128-multilayer-output-2-6