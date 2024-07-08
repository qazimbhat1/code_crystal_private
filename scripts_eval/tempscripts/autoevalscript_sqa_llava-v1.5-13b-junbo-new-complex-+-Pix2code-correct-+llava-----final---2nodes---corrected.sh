#!/bin/bash 
#SBATCH --job-name=a_sqallava-v1.5-13b-junbo-new-complex-+-Pix2code-correct-+llava-----final---2nodes---corrected
#SBATCH --output=eval_sqa_llava-v1.5-13b-junbo-new-complex-+-Pix2code-correct-+llava-----final---2nodes---corrected.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-13b-junbo-new-complex-+-Pix2code-correct-+llava-----final---2nodes---corrected \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-junbo-new-complex-+-Pix2code-correct-+llava-----final---2nodes---corrected02.01-17.34.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-junbo-new-complex-+-Pix2code-correct-+llava-----final---2nodes---corrected02.01-17.34.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-junbo-new-complex-+-Pix2code-correct-+llava-----final---2nodes---corrected02.01-17.34.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-junbo-new-complex-+-Pix2code-correct-+llava-----final---2nodes---corrected02.01-17.34.json