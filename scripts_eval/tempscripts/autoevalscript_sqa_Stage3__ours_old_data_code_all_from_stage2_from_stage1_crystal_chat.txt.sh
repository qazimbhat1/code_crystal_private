#!/bin/bash 
#SBATCH --job-name=a_sqaStage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt
#SBATCH --output=eval_sqa_Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt02.26-19.44.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt02.26-19.44.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt02.26-19.44.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt02.26-19.44.json