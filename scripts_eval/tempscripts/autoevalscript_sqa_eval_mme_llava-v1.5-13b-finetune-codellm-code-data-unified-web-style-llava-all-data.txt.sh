#!/bin/bash 
#SBATCH --job-name=a_sqaeval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt
#SBATCH --output=eval_sqa_eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt05.08-14.34.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt05.08-14.34.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt05.08-14.34.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/eval_mme_llava-v1.5-13b-finetune-codellm-code-data-unified-web-style-llava-all-data.txt05.08-14.34.json