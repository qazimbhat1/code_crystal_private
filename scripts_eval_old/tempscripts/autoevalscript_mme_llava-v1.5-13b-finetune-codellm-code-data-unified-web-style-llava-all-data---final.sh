#!/bin/bash 
#SBATCH --job-name=a_mmellava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final
#SBATCH --output=eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_llava/playground/data/eval/MME/answers/llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final01.02-17.09.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final01.02-17.09

cd eval_tool 

python calculation.py --results_dir answers/llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data---final01.02-17.09