#!/bin/bash 
#SBATCH --job-name=a_sqastage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat
#SBATCH --output=eval_sqa_stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat03.14-16.34.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat03.14-16.34.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat03.14-16.34.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat03.14-16.34.json