#!/bin/bash 
#SBATCH --job-name=a_mmeStage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt
#SBATCH --output=eval_mme_Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15

cd eval_tool 

python calculation.py --results_dir answers/Stage3____finetune_Pretrained_with_short_caption______+_______crystal_chat_tokenizer_correct_Websight_websrc_code_QA_llava.txt02.18-18.15