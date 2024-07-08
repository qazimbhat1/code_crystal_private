#!/bin/bash 
#SBATCH --job-name=a_mmellava-v1.5-stage3_C1_2_epochs_new_data_old_projector_from_stage2_from_stage1_crystal_chat
#SBATCH --output=eval_mme_llava-v1.5-stage3_C1_2_epochs_new_data_old_projector_from_stage2_from_stage1_crystal_chat.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/llava-v1.5-stage3_C1_2_epochs_new_data_old_projector_from_stage2_from_stage1_crystal_chat \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/llava-v1.5-stage3_C1_2_epochs_new_data_old_projector_from_stage2_from_stage1_crystal_chat03.01-18.59.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava-v1.5-stage3_C1_2_epochs_new_data_old_projector_from_stage2_from_stage1_crystal_chat03.01-18.59

cd eval_tool 

python calculation.py --results_dir answers/llava-v1.5-stage3_C1_2_epochs_new_data_old_projector_from_stage2_from_stage1_crystal_chat03.01-18.59