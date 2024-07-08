#!/bin/bash 
#SBATCH --job-name=a_mmellava-v1.5-7b-85_15_split_same_as_codellm
#SBATCH --output=eval_mme_llava-v1.5-7b-85_15_split_same_as_codellm.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-7b-85_15_split_same_as_codellm \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/llava-v1.5-7b-85_15_split_same_as_codellm01.03-09.32.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava-v1.5-7b-85_15_split_same_as_codellm01.03-09.32

cd eval_tool 

python calculation.py --results_dir answers/llava-v1.5-7b-85_15_split_same_as_codellm01.03-09.32