#!/bin/bash 
#SBATCH --job-name=a_sqallava-v1.5-LLaVA_finetuning_using_ours_crystal_chat_using_pretrain_with_ours___LLaVA_data
#SBATCH --output=eval_sqa_llava-v1.5-LLaVA_finetuning_using_ours_crystal_chat_using_pretrain_with_ours___LLaVA_data.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-LLaVA_finetuning_using_ours_crystal_chat_using_pretrain_with_ours___LLaVA_data \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-LLaVA_finetuning_using_ours_crystal_chat_using_pretrain_with_ours___LLaVA_data03.05-22.06.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-LLaVA_finetuning_using_ours_crystal_chat_using_pretrain_with_ours___LLaVA_data03.05-22.06.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-LLaVA_finetuning_using_ours_crystal_chat_using_pretrain_with_ours___LLaVA_data03.05-22.06.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-LLaVA_finetuning_using_ours_crystal_chat_using_pretrain_with_ours___LLaVA_data03.05-22.06.json