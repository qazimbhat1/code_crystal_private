#!/bin/bash 
#SBATCH --job-name=a_textvqallava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data
#SBATCH --output=eval_textvqa_llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data.txt
#SBATCH --nodes=1
#SBATCH --mem=490G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4 -p gpumid
#SBATCH --reservation=vision


CUDA_VISIBLE_DEVICES=0,1,2,3 
        
python -m llava.eval.model_vqa_loader \
    --model-path /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/LLaVA/checkpoints/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data02.01-14.41.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-WebSRC+Unified+WebSight+Code+LLaVA-data_+_150k_QA_data02.01-14.41.jsonl