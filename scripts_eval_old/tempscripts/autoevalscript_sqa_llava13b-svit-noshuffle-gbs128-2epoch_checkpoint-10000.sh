#!/bin/bash 
#SBATCH --job-name=a_sqallava13b-svit-noshuffle-gbs128-2epoch_checkpoint-10000
#SBATCH --output=eval_sqa_llava13b-svit-noshuffle-gbs128-2epoch_checkpoint-10000.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path checkpoints/llava13b-svit-noshuffle-gbs128-2epoch/checkpoint-10000 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava13b-svit-noshuffle-gbs128-2epoch_checkpoint-1000012.22-17.15.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava13b-svit-noshuffle-gbs128-2epoch_checkpoint-1000012.22-17.15.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava13b-svit-noshuffle-gbs128-2epoch_checkpoint-1000012.22-17.15.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava13b-svit-noshuffle-gbs128-2epoch_checkpoint-1000012.22-17.15.json