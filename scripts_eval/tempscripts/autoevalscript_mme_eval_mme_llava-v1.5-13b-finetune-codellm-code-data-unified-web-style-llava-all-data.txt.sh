#!/bin/bash 
#SBATCH --job-name=a_mmeeval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt
#SBATCH --output=eval_mme_eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt05.08-14.34.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt05.08-14.34

cd eval_tool 

python calculation.py --results_dir answers/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt05.08-14.34