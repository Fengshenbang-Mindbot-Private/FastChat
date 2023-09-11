#!/bin/bash

#SBATCH --job-name=fastchat_general # create a short name for your job
#SBATCH --nodes=2 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=48G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:8 # number of gpus per node
#SBATCH -p pot # number of gpus per node

#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/%x-%j.err # output and error log file names (%x for job id)

# pot-preempted
# --warmup_ratio 0.03

NNODES=2
GPUS_PER_NODE=8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)

WANDB_PROJECT="FastChat"
export WANDB_PROJECT

export LAUNCHER="torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

export CMD=" train.py \
    --model_name_or_path /cognitive_comp/pankunhao/pretrained/Ziya-LLaMa2-13B-step115000 \
    --data_path /cognitive_comp/pankunhao/data/writing/general_sft_data/train.json \
    --eval_data_path /cognitive_comp/pankunhao/data/writing/general_sft_data/eval.json \
    --bf16 True \
    --output_dir /cognitive_comp/pankunhao/code/FastChat/model_ckpt/general_0906 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.15 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --deepspeed "ds_config.json" \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
"

echo ${CMD}

srun bash -c '$LAUNCHER --node_rank=$SLURM_PROCID $CMD'
