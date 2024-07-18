#!/bin/bash 
#SBATCH --job-name=a_sqastage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat
#SBATCH --output=eval_sqa_stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat03.05-19.13.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat03.05-19.13.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat03.05-19.13.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat03.05-19.13.json