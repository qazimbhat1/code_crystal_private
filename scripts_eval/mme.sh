#!/bin/bash
#SBATCH --job-name=mme_new # Job name
#SBATCH --output=mme-new.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final
cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final