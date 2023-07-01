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
from transformers import PreTrainedModel, PreTrainedTokenizer


def load_and_cache_examples(args: Config, tokenizer, df: DataFrame):
    return ConversationDataset(tokenizer, args, df)


def set_seed(args: Config):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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