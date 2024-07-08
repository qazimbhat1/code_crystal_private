#!/bin/bash 
#SBATCH --job-name=a_mmeSTAGE_2_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption
#SBATCH --output=eval_mme_STAGE_2_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/checkpoints/STAGE_2_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/STAGE_2_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption02.14-16.33.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment STAGE_2_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption02.14-16.33

cd eval_tool 

python calculation.py --results_dir answers/STAGE_2_Long_Caption_no_mimic_shuffled_stage_1_shuffled_short_caption02.14-16.33