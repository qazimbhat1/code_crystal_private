#!/bin/bash 
#SBATCH --job-name=a_sqallava-v1.5-finetune-crystal_chat-tokenizer-correct_Websight_websrc_code_QA_llava
#SBATCH --output=eval_sqa_llava-v1.5-finetune-crystal_chat-tokenizer-correct_Websight_websrc_code_QA_llava.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-finetune-crystal_chat-tokenizer-correct_Websight_websrc_code_QA_llava \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-finetune-crystal_chat-tokenizer-correct_Websight_websrc_code_QA_llava01.31-16.45.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-finetune-crystal_chat-tokenizer-correct_Websight_websrc_code_QA_llava01.31-16.45.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-finetune-crystal_chat-tokenizer-correct_Websight_websrc_code_QA_llava01.31-16.45.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/llava-v1.5-finetune-crystal_chat-tokenizer-correct_Websight_websrc_code_QA_llava01.31-16.45.json