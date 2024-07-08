#!/bin/bash
#SBATCH --job-name=evalsqa # Job name
#SBATCH --output=sqa_70blora.txt
#SBATCH --nodes=1
#SBATCH --nodelist=gpumid-17
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

CUDA_VISIBLE_DEVICES=0,1,2,3

python llava/eval/model_vqa_science.py \
    --model-path ./checkpoints/llava-v1.5-70blora-bs2as64-finetuned \
    --model-base ./pretweight/llama2_70b_hfweight \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-70blora-bs2as64-finetuned.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-70blora-bs2as64-finetuned.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-70blora-bs2as64-finetuned_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-70blora-bs2as64-finetuned_result.json
