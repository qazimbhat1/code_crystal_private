#!/bin/bash 
#SBATCH --job-name=a_popeStage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt
#SBATCH --output=eval_pope_Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/coco \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/pope/answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15.jsonl
