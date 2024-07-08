#!/bin/bash 
#SBATCH --job-name=a_textvqallava-13b-lvis880k-2epo-gbs128
#SBATCH --output=eval_textvqa_llava-13b-lvis880k-2epo-gbs128.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 
        
python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/lrvset/llava-13b-lvis880k-2epo-gbs128 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-13b-lvis880k-2epo-gbs12812.27-18.24.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-13b-lvis880k-2epo-gbs12812.27-18.24.jsonl