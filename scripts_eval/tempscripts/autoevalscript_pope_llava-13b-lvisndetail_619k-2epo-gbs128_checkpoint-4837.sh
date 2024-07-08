#!/bin/bash 
#SBATCH --job-name=a_popellava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-4837
#SBATCH --output=eval_pope_llava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-4837.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/lrvset/llava-13b-lvisndetail_619k-2epo-gbs128/checkpoint-4837 \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-483712.26-14.00.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-483712.26-14.00.jsonl
