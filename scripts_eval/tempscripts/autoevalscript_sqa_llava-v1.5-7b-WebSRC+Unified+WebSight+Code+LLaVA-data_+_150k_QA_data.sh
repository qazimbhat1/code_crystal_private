#!/bin/bash 
#SBATCH --job-name=a_sqallava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data
#SBATCH --output=eval_sqa_llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision

python llava/eval/model_vqa_science.py \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data02.01-14.41.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data02.01-14.41.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data02.01-14.41.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data02.01-14.41.json