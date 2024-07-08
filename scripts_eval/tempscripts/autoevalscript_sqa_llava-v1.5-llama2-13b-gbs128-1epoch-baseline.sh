#!/bin/bash 
#SBATCH --job-name=a_sqallava-v1.5-llama2-13b-gbs128-1epoch-baseline
#SBATCH --output=eval_sqa_llava-v1.5-llama2-13b-gbs128-1epoch-baseline.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path checkpoints/llava-v1.5-llama2-13b-gbs128-1epoch-baseline \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-llama2-13b-gbs128-1epoch-baseline12.28-17.30.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-llama2-13b-gbs128-1epoch-baseline12.28-17.30.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-llama2-13b-gbs128-1epoch-baseline12.28-17.30.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-llama2-13b-gbs128-1epoch-baseline12.28-17.30.json