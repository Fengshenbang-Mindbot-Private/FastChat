#!/bin/bash

#SBATCH --job-name=fastchat_eval # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=48G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:8 # number of gpus per node
#SBATCH -p pot-preempted # number of gpus per node

#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/%x-%j.err # output and error log file names (%x for job id)

# pot-preempted

NNODES=1
GPUS_PER_NODE=8

source activate base

python get_model_answer.py \
    --model-path /cognitive_comp/pankunhao/pretrained/baichuan-vicuna-chinese-7b/ \
    --model-id "baichuan" \
    --question-file /cognitive_comp/pankunhao/data/writing/eval_data/general_raw.jsonl \
    --answer-file /cognitive_comp/pankunhao/data/writing/eval_data/general_ans_baichuan.jsonl \
    --num-gpus ${GPUS_PER_NODE}
