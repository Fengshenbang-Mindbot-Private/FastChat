#!/bin/bash

#SBATCH --job-name=fastchat_writing # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:8 # number of gpus per node
#SBATCH -p pot # number of gpus per node

#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/%x-%j.err # output and error log file names (%x for job id)

# pot-preempted
# --warmup_ratio 0.03

NNODES=1
GPUS_PER_NODE=8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)

WANDB_PROJECT="FastChat"
export WANDB_PROJECT

torchrun --nproc_per_node=$GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py \
    --model_name_or_path "/cognitive_comp/pankunhao/code/FastChat/model_ckpt/general_0906" \
    --data_path /cognitive_comp/pankunhao/data/writing/sft_data/train_inverse.json \
    --eval_data_path /cognitive_comp/pankunhao/data/writing/sft_data/eval_inverse.json \
    --bf16 True \
    --output_dir /cognitive_comp/pankunhao/code/FastChat/model_ckpt/inverse_0908 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed "ds_config.json" \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True
