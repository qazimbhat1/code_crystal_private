#!/bin/bash 
#SBATCH --job-name=a_popellava-v1.5-llama2-13b-gbs128-1epoch-baseline
#SBATCH --output=eval_pope_llava-v1.5-llama2-13b-gbs128-1epoch-baseline.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava-v1.5-llama2-13b-gbs128-1epoch-baseline \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-llama2-13b-gbs128-1epoch-baseline12.28-17.30.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-llama2-13b-gbs128-1epoch-baseline12.28-17.30.jsonl
