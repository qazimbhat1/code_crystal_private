#!/bin/bash
#SBATCH --job-name=mmeeval # Job name
#SBATCH --output=mme_70b.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

CUDA_VISIBLE_DEVICES=0,1,2,3

python -m llava.eval.model_vqa_loader \
    --model-path ./running_result/pretrained_models/llava-v1.5-70blora-onlypt \
    --model-base ./pretweight/llama2_70b_hfweight \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-70b-onlypt.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-70b-onlypt

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-70b-onlypt