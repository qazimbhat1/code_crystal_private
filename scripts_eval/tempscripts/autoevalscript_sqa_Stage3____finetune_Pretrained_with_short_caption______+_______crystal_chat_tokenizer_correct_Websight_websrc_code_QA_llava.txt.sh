#!/bin/bash 
#SBATCH --job-name=a_sqaStage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt
#SBATCH --output=eval_sqa_Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/test \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15.jsonl \
    --output-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15.jsonl \
    --output-result /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/scienceqa/answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15.json