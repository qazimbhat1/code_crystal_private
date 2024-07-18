#!/bin/bash 
#SBATCH --job-name=a_popestage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat
#SBATCH --output=eval_pope_stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/answers/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat03.05-19.13.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/coco \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/answers/stage3_new_data_shuffled_+_Stage1_LLaVA_pretrain_data______+_______crystal_chat03.05-19.13.jsonl
