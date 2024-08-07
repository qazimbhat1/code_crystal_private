#!/bin/bash 
#SBATCH --job-name=a_mmellava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_15
#SBATCH --output=eval_mme_llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_15.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_15 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/MME/answers/llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_1512.24-14.19.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/MME 

python convert_answer_to_mme.py --experiment llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_1512.24-14.19

cd eval_tool 

python calculation.py --results_dir answers/llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_1512.24-14.19