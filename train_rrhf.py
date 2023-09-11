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
#

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import random

# Need to call this before importing transformers.
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

import torch
import torch.nn.functional as F
import transformers
import wandb
from transformers import Trainer
from transformers import DefaultDataCollator
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.utils import rank0_print
from fastchat.data.rrhf_dataset import RRHFDataset, LazyRRHFDataset

local_rank = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
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
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_position_interpolation: Optional[bool] = field(default=False)
    report_to: str = field(default="wandb")
    rrhf_weight: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    only_use_provide: bool = field(default=False)
    only_use_sample: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazyRRHFDataset if data_args.lazy_preprocess else RRHFDataset
    )
    rank0_print("Loading data...")

    train_json = []
    for line in open(data_args.data_path):
        train_json.append(json.loads(line))
    # train_json = json.load(open(data_args.data_path, "r"))
    random.shuffle(train_json)
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        
        from pathlib import Path
        eval_dataset = {}
        for eval_data_path in data_args.eval_data_path:
            dataset_name = Path(eval_data_path).stem
            rank0_print(f"Loading {dataset_name} eval data...")
            # eval_json = json.load(open(eval_data_path, "r"))
            eval_json = []
            for line in open(eval_data_path):
                eval_json.append(json.loads(line))
            eval_dataset[dataset_name] = RRHFDataset(eval_json, tokenizer=tokenizer)
        
        # eval_dataset = dataset_cls(train_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    data_collator = DefaultDataCollator()
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

class RRHFTrainer(Trainer):
    def gather_logits_labels(self, logits, labels):

        mask = (labels != LabelSmoother.ignore_index).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == LabelSmoother.ignore_index] = 0 
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != LabelSmoother.ignore_index).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, idxs, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1) # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1) # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        # rank0_print("scores: ", scores)
        # rank0_print("rw_diff: ", rw_diff)
        # rank0_print("bitwise_and: ", torch.bitwise_and(rw_diff > 0, diff < 0))
        # rank0_print("diff:", diff)
        # rank0_print("diff shape:", diff.shape)
        # rank0_print("aval:", aval)
        # rank0_print("aval shape:", aval.shape)
        return -diff[aval].sum()

    def sft_loss(self, logit_label, idxs, rw_scores):
        max_idx = torch.argmax(rw_scores)
        return -logit_label[max_idx].mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        # rank0_print("inputs: ", inputs["input_ids"].shape)
        input_ids = inputs["input_ids"].squeeze(0)
        # rank0_print("input_ids: ", input_ids.shape)
        attention_mask = inputs["attention_mask"].squeeze(0)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)[0] # (batch * cand) * L * V
        # rank0_print("logits: ", logits.shape)
        logits = F.log_softmax(logits, dim=-1)
        # rank0_print("logits: ", logits.shape)
        labels = inputs["labels"].squeeze(0)
        # rank0_print("labels: ", labels.shape)
        logit_label = self.gather_logits_labels(logits, labels)
        # rank0_print("logit_label:", logit_label.shape)
        scores = self.get_score(logit_label, labels)
        # rank0_print("scores:", scores)
        rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
        sft_loss = self.sft_loss(logit_label, inputs.get("idxs"), inputs.get("scores"))
        rank0_print("rrhf_loss: ", rrhf_loss)
        rank0_print("sft_loss: ", sft_loss)
        wandb.log({"rrhf_loss": rrhf_loss}, step=self.global_step)
        wandb.log({"sft_loss": sft_loss}, step=self.global_step)
        loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    from tokenizers import AddedToken
    human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
    bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [human, bot]})
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = RRHFTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
