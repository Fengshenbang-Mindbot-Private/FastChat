#encoding=utf8

import sys
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.model.model_adapter import get_conversation_template
from fastchat.utils import rank0_print
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def _single_tokenize(text, tokenizer, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length
    toked = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        )
    return toked['input_ids'][0]

def tokenize(
    conv,
    conversations,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            torch.set_printoptions(threshold=10000)
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print("readable target:", tokenizer.decode(z))
            rank0_print("conversation:", conversation)
            rank0_print("input_ids:", input_ids[idx])
            rank0_print("target:", target)

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
        conv = get_conversation_template("ziya_13b")

        idxs = []
        all_scores = []
        input_ids = []
        labels = []
        for idx, ins in enumerate(sources):
            query = ins["prompt"]
            responses = ins["answers"]
            scores = ins["rewards"]
            all_scores.append(scores)
            idxs.append([idx] * len(scores))
            cand_input_ids = []
            cand_labels = []
            conversations = []
            for res in responses:
                conv.messages = []
                conv.append_message(conv.roles[0], query)
                conv.append_message(conv.roles[1], res)
                conversations.append(conv.get_prompt())
            res_ret = tokenize(conv, conversations, tokenizer)
            cand_input_ids = res_ret["input_ids"]
            cand_labels = res_ret["labels"]
            input_ids.append(cand_input_ids)
            labels.append(cand_labels)

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )

class RRHFDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(RRHFDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.scores = data_dict["scores"]
        self.idxs = data_dict["idxs"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i],
                    idxs=self.idxs[i],
                    scores=self.scores[i]
        )


class LazyRRHFDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazyRRHFDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        # print("lazy ret input_ids:", ret["input_ids"].shape)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            idxs=ret["idxs"][0],
            scores=ret["scores"][0],
        )
        # print("lazy dict input_ids:", ret["input_ids"].shape)
        self.cached_data_dict[i] = ret
        return ret
