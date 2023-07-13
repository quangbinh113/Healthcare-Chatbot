from config import Config
from utils import set_seed, _sorted_checkpoints
import os
import glob
import argparse

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead,
)
import torch
from train import train, evaluate 
import pandas as pd
from pandas import DataFrame

def main(args: Config):
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args)

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)

    print("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train(args, model, tokenizer)

    # Evaluation
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
                        
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default="DialoGPT-small")
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--train_path', type=str, default="../data/train_.csv")
    parser.add_argument('--val_path', type=str, default="../data/train_.csv")
    
    parse = parser.parse_args()
    
    data_path = {
                    'train': parse.train_path,
                    'val': parse.val_path
                }
    args = Config()
    
    args.data_path = data_path
    args.num_train_epochs = parse.num_epochs
    args.experiment_name = parse.experiment_name
    args.name = parse.model
    args.model_name_or_path = 'microsoft/' + parse.model
    args.config_name = 'microsoft/' + parse.model
    args.tokenizer_name = 'microsoft/' + parse.model
    
    main(args)
    print("Done!")