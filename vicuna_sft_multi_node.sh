#!/bin/bash

#SBATCH --job-name=fastchat_multinode # create a short name for your job
#SBATCH --nodes=2 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -x dgx047
#SBATCH --reservation=acagpt

#SBATCH -o ./log/%x-%j.log # output and error log file names (%x for job id)
#SBATCH -e ./log/%x-%j.err # output and error log file names (%x for job id)


NNODES=2
GPUS_PER_NODE=8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)

export LAUNCHER="torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

FSDP_STRATEGY="full_shard auto_wrap"

export CMD=" train.py \
    --model_name_or_path /cognitive_comp/pankunhao/pretrained/pytorch/llama-13b \
    --data_path ./playground/data/dummy.json \
    --bf16 True \
    --output_dir /cognitive_comp/pankunhao/code/FastChat/output \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
"

echo ${CMD}

srun bash -c '$LAUNCHER --node_rank=$SLURM_PROCID $CMD --fsdp "full_shard offload auto_wrap"'
