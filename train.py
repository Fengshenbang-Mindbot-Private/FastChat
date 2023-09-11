# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import random
import pathlib
from typing import Dict, Optional, Sequence

# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

import torch
import transformers
from transformers import Trainer
import wandb

from fastchat.utils import rank0_print
from fastchat.data.sft_dataset import SupervisedDataset, LazySupervisedDataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data.", "nargs": "*"}
        # default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_position_interpolation: Optional[bool] = field(default=False)
    report_to: str = field(default="wandb")


local_rank = None

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    random.shuffle(train_json)
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        
        from pathlib import Path
        eval_dataset = {}
        for eval_data_path in data_args.eval_data_path:
            dataset_name = Path(eval_data_path).stem
            rank0_print(f"Loading {dataset_name} eval data...")
            eval_json = json.load(open(eval_data_path, "r"))
            eval_dataset[dataset_name] = SupervisedDataset(eval_json, tokenizer=tokenizer)
        
        # eval_dataset = dataset_cls(train_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    # Set RoPE scaling factor
    if training_args.use_position_interpolation:
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    from tokenizers import AddedToken
    human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
    bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [human, bot]})
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("resume from checkpoint.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    model.config.use_cache = True
    
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
