#!/bin/bash 
#SBATCH --job-name=a_mmellava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-4837
#SBATCH --output=eval_mme_llava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-4837.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/lrvset/llava-13b-lvisndetail_619k-2epo-gbs128/checkpoint-4837 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-483712.26-14.00.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-483712.26-14.00

cd eval_tool 

python calculation.py --results_dir answers/llava-13b-lvisndetail_619k-2epo-gbs128_checkpoint-483712.26-14.00