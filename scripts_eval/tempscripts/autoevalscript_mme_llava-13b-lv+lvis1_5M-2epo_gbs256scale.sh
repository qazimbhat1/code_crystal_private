#!/bin/bash 
#SBATCH --job-name=a_mmellava-13b-lv+lvis1_5M-2epo_gbs256scale
#SBATCH --output=eval_mme_llava-13b-lv+lvis1_5M-2epo_gbs256scale.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/lrvset/llava-13b-lv+lvis1_5M-2epo_gbs256scale \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-13b-lv+lvis1_5M-2epo_gbs256scale12.29-20.29.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava-13b-lv+lvis1_5M-2epo_gbs256scale12.29-20.29

cd eval_tool 

python calculation.py --results_dir answers/llava-13b-lv+lvis1_5M-2epo_gbs256scale12.29-20.29