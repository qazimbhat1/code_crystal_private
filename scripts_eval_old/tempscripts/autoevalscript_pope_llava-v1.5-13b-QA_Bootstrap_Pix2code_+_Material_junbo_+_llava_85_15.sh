#!/bin/bash 
#SBATCH --job-name=a_popellava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_15
#SBATCH --output=eval_pope_llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_15.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_15 \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_1512.24-13.16.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b-QA_Bootstrap_Pix2code_+_Material_junbo_+_llava_85_1512.24-13.16.jsonl
