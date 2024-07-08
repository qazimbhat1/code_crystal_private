#!/bin/bash
#SBATCH --job-name=popeeval # Job name
#SBATCH --output=pope_70blora.txt
#SBATCH --nodes=1
#SBATCH --nodelist=gpumid-11
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

CUDA_VISIBLE_DEVICES=0,1,2,3

python llava/eval/model_vqa_loader.py \
    --model-path ./checkpoints/llava-v1.5-70blora-bs2as64-finetuned \
    --model-base ./pretweight/llama2_70b_hfweight \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-70blora-bs2as64-finetuned.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-70blora-bs2as64-finetuned.jsonl
