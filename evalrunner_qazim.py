import os
import argparse
from datetime import datetime

def clean_ckpt_name(ckptname):
    if ckptname[-1] == "/": ckptname = ckptname[:-1]
    if "checkpoint-" in ckptname:
        ckptname = ckptname.split("/")[-2] + "_" + ckptname.split("/")[-1]
    else:
        ckptname = ckptname.split("/")[-1]
    
    return ckptname

def eval_handler(args):
    runner = eval_commands()
    funclist = [runner.mme, runner.pope, runner.sqa, runner.textvqa]
    funcname = ["mme", "pope", "sqa", "textvqa"]
    paramlist = [args.mme, args.pope, args.sqa, args.textvqa]
    ckptname = clean_ckpt_name(args.ckpt)

    for i in range(len(funclist)):
        if paramlist[i] or args.all:
            slurm_head = "#!/bin/bash \n\
#SBATCH --job-name=%s\n\
#SBATCH --output=%s.txt\n\
#SBATCH --nodes=1\n\
#SBATCH --mem=490G\n\
#SBATCH --cpus-per-task=64\n\
#SBATCH --gres=gpu:4 -p gpumid\n\
#SBATCH --reservation=vision\n\n"%("a_"+funcname[i]+ckptname,"eval_"+funcname[i]+"_"+ckptname)

            slurm_head += funclist[i](args.ckpt)

            script_dir = "./scripts_eval/tempscripts/"+"autoevalscript_"+funcname[i]+"_"+ckptname+".sh"

            with open(script_dir, "w") as f:
                f.write(slurm_head)

            # Run the sh script
            print("Submitted %s evaluation for metric %s:"%(ckptname, funcname[i]))
            os.system("sbatch "+script_dir)
    

    
class eval_commands:
    def fr(self,runname):
        rnnew = clean_ckpt_name(runname)
        return rnnew+datetime.now().strftime("%m.%d-%H.%M")
    
    def mme(self,runname):
        return "python -m llava.eval.model_vqa_loader \\\n\
    --model-path %s \\\n\
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \\\n\
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \\\n\
    --answers-file /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME/answers/%s.jsonl \\\n\
    --temperature 0 \\\n\
    --conv-mode vicuna_v1\n\
\n\
cd /lustre/scratch/shared-folders/vision-project/Code/qazim.bhat/fork_LLaVA/playground/data/eval/MME \n\
\n\
python convert_answer_to_mme.py --experiment %s\n\n\
cd eval_tool \n\
\n\
python calculation.py --results_dir answers/%s"%(runname, self.fr(runname), self.fr(runname), self.fr(runname))
    
    def pope(self,runname):
        return "\nCUDA_VISIBLE_DEVICES=0,1,2,3 \n\
\n\
python -m llava.eval.model_vqa_loader \\\n\
    --model-path %s \\\n\
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \\\n\
    --image-folder ./playground/data/eval/pope/val2014 \\\n\
    --answers-file ./playground/data/eval/pope/answers/%s.jsonl \\\n\
    --temperature 0 \\\n\
    --conv-mode vicuna_v1\n\
\n\
python llava/eval/eval_pope.py \\\n\
    --annotation-dir ./playground/data/eval/pope/coco \\\n\
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \\\n\
    --result-file ./playground/data/eval/pope/answers/%s.jsonl\n\
"%(runname, self.fr(runname), self.fr(runname))
    
    def sqa(self,runname):
        return "python llava/eval/model_vqa_science.py \\\n\
    --model-path %s \\\n\
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \\\n\
    --image-folder ./playground/data/eval/scienceqa/test \\\n\
    --answers-file ./playground/data/eval/scienceqa/answers/%s.jsonl \\\n\
    --single-pred-prompt \\\n\
    --temperature 0 \\\n\
    --conv-mode vicuna_v1\n\
\n\
python llava/eval/eval_science_qa.py \\\n\
    --base-dir ./playground/data/eval/scienceqa \\\n\
    --result-file ./playground/data/eval/scienceqa/answers/%s.jsonl \\\n\
    --output-file ./playground/data/eval/scienceqa/answers/%s.jsonl \\\n\
    --output-result ./playground/data/eval/scienceqa/answers/%s.json"%(runname, self.fr(runname), self.fr(runname), self.fr(runname), self.fr(runname))
    
    def textvqa(self, runname):
        return "\nCUDA_VISIBLE_DEVICES=0,1,2,3 \n\
        \n\
python -m llava.eval.model_vqa_loader \\\n\
    --model-path %s \\\n\
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \\\n\
    --image-folder ./playground/data/eval/textvqa/train_images \\\n\
    --answers-file ./playground/data/eval/textvqa/answers/%s.jsonl \\\n\
    --temperature 0 \\\n\
    --conv-mode vicuna_v1 \n\
\n\
python -m llava.eval.eval_textvqa \\\n\
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \\\n\
    --result-file ./playground/data/eval/textvqa/answers/%s.jsonl"%(runname, self.fr(runname), self.fr(runname))


if __name__ == "__main__":
    # receive args
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", type=str, default=None)
    parser.add_argument("--all", "-a", action="store_true", default=False)
    parser.add_argument("--mme", action="store_true", default=False)
    parser.add_argument("--pope", action="store_true", default=False)
    parser.add_argument("--sqa", action="store_true", default=False)
    parser.add_argument("--textvqa", action="store_true", default=False)
    args = parser.parse_args()

    eval_handler(args)