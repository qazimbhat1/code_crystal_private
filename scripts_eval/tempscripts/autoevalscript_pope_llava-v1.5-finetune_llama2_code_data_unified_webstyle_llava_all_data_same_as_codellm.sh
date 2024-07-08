#!/bin/bash 
#SBATCH --job-name=a_popellava-v1.5-finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm
#SBATCH --output=eval_pope_llava-v1.5-finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/answers/llava-v1.5-finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm03.12-10.58.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/coco \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/answers/llava-v1.5-finetune_llama2_code_data_unified_webstyle_llava_all_data_same_as_codellm03.12-10.58.jsonl