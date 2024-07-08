#!/bin/bash 
#SBATCH --job-name=a_mmellava13b-svit-noshuffle-gbs128-2epoch
#SBATCH --output=eval_mme_llava13b-svit-noshuffle-gbs128-2epoch.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava13b-svit-noshuffle-gbs128-2epoch \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava13b-svit-noshuffle-gbs128-2epoch12.25-17.29.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava13b-svit-noshuffle-gbs128-2epoch12.25-17.29

cd eval_tool 

python calculation.py --results_dir answers/llava13b-svit-noshuffle-gbs128-2epoch12.25-17.29