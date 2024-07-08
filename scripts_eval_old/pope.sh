#!/bin/bash
#SBATCH --job-name=popeeval # Job name
#SBATCH --output=pope_subsvit7000in2epo.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

CUDA_VISIBLE_DEVICES=0,1,2,3

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava13b-subsvit-noshuffle-gbs128-2epoch/checkpoint-7000 \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava13b-subsvit-noshffule-gbs128-7000in2eporesult.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava13b-subsvit-noshffule-gbs128-7000in2eporesult.jsonl
