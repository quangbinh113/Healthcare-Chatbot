from dataset import ConversationDataset 
import torch
import random
import numpy as np
from config import Config
import shutil
import glob
import os
from typing import List
import re
from pandas import DataFrame
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer



def load_and_cache_examples(args: Config, tokenizer, df: DataFrame):
    return ConversationDataset(tokenizer, args, df)


def set_seed(args: Config):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def read_file(path: dict):
    df_train = pd.read_csv(path['train'])
    df_train = df_train.drop(columns='Unnamed: 0')
    df_val = pd.read_csv(path['val'])
    df_val = df_val.drop(columns='Unnamed: 0')
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    return df_train, df_val

def save_model(args: Config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, optimizer, scheduler):
    output_dir = os.path.join(args.output_dir, f"{args.name}_best_model")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    print(f"Saving the best model checkpoint to {output_dir}")

    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    # print(f"Saving optimizer and scheduler states to {output_dir}")