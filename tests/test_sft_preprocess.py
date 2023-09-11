#encoding=utf8

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import numpy as np
import torch
torch.set_printoptions(threshold=np.inf)
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.data.sft_dataset import preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.unk_token

    raw_data = json.load(open(args.input_file, "r"))
    print("load %d raw data" % len(raw_data))
    sources = [example["conversations"] for example in raw_data]
    sources = sources[:10]
    data_dict = preprocess(sources, tokenizer)
