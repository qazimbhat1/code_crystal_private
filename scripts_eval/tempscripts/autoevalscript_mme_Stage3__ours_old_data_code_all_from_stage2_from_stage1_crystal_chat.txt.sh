#!/bin/bash 
#SBATCH --job-name=a_mmeStage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt
#SBATCH --output=eval_mme_Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt02.26-19.44.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt02.26-19.44

cd eval_tool 

python calculation.py --results_dir answers/Stage3__ours_old_data_code_all_from_stage2_from_stage1_crystal_chat.txt02.26-19.44