#encoding=utf8

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import torch
import random
torch.set_printoptions(threshold=np.inf)
import transformers

from fastchat.data.rrhf_dataset import preprocess

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

    sources = []
    for line in open(args.input_file, "r"):
        data = json.loads(line.strip())
        sources.append(data)
    print("load %d raw data" % len(sources))
    random.shuffle(sources)
    sources = sources[:1]
    data_dict = preprocess(sources, tokenizer)
