#!/bin/bash 
#SBATCH --job-name=a_popellava-v1.5-13b-OCR-10k-junbo-new-complex-+-Pix2code-correct-+llava-----final
#SBATCH --output=eval_pope_llava-v1.5-13b-OCR-10k-junbo-new-complex-+-Pix2code-correct-+llava-----final.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 

python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-13b-OCR-10k-junbo-new-complex-+-Pix2code-correct-+llava-----final \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b-OCR-10k-junbo-new-complex-+-Pix2code-correct-+llava-----final06.05-19.42.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/coco \
    --question-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/playground/data/eval/pope/answers/llava-v1.5-13b-OCR-10k-junbo-new-complex-+-Pix2code-correct-+llava-----final06.05-19.42.jsonl