#!/bin/bash 
#SBATCH --job-name=a_mmestage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat
#SBATCH --output=eval_mme_stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat03.14-16.34.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat03.14-16.34

cd eval_tool 

python calculation.py --results_dir answers/stage3_with_text_mlp_from_stage_2_with_text_mlp_from_stage_1_mlp_crystal_chat03.14-16.34