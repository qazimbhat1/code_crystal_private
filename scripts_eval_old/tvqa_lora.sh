#!/bin/bash
#SBATCH --job-name=tvqaeval # Job name
#SBATCH --output=textvqa_70blora.txt
#SBATCH --nodes=1
#SBATCH --nodelist=gpumid-17
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

CUDA_VISIBLE_DEVICES=0,1,2,3

python llava/eval/model_vqa_loader.py \
    --model-path ./checkpoints/llava-v1.5-70blora-bs2as64-finetuned \
    --model-base ./pretweight/llama2_70b_hfweight \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-70blora-bs2as64-finetuned.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-70blora-bs2as64-finetuned.jsonl
