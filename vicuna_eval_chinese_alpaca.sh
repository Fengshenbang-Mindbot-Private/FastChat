#!/bin/bash

#SBATCH --job-name=fastchat_eval # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=48G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:4 # number of gpus per node
#SBATCH -p pot # number of gpus per node

#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/%x-%j.err # output and error log file names (%x for job id)

# pot-preempted

NNODES=1
GPUS_PER_NODE=4

source activate base

python get_model_answer.py \
    --model-path /cognitive_comp/pankunhao/pretrained/chinese-alpaca-plus-13b-hf/ \
    --model-id "chinese_alpaca" \
    --question-file /cognitive_comp/pankunhao/data/writing/eval_data/writing_sbs.jsonl \
    --answer-file /cognitive_comp/pankunhao/data/writing/eval_data/writing_sbs_ans_chinese_alpaca.jsonl \
    --num-gpus ${GPUS_PER_NODE}
