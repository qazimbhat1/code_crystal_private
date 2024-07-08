#!/bin/bash 
#SBATCH --job-name=a_sqaSTAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption
#SBATCH --output=eval_sqa_STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption02.14-16.36.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption02.14-16.36.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption02.14-16.36.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/STAGE_2_FREEZING_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption02.14-16.36.json