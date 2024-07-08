#!/bin/bash 
#SBATCH --job-name=a_sqallava-13b-lvis880k-2epo-gbs128
#SBATCH --output=eval_sqa_llava-13b-lvis880k-2epo-gbs128.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path checkpoints/lrvset/llava-13b-lvis880k-2epo-gbs128 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-13b-lvis880k-2epo-gbs12812.27-18.24.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-13b-lvis880k-2epo-gbs12812.27-18.24.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-13b-lvis880k-2epo-gbs12812.27-18.24.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-13b-lvis880k-2epo-gbs12812.27-18.24.json