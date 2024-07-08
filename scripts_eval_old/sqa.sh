#!/bin/bash
#SBATCH --job-name=evalsqa # Job name
#SBATCH --output=sqa_13btuneclip.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

### python -m llava.eval.model_vqa_science \
python llava/eval/model_vqa_science.py \
    --model-path checkpoints/llava13b_bestconfig_tuneclip \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava13b_bestconfig_tuneclip.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava13b_bestconfig_tuneclip.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava13b_bestconfig_tuneclip_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava13b_bestconfig_tuneclip_result.json
