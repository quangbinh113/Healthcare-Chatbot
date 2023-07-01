from config import Config
from utils import set_seed, _sorted_checkpoints
import os
import glob
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead,
)
import torch
from train import train, evaluate 
import pandas as pd
from pandas import DataFrame

def main(args: Config, df_train: DataFrame, df_val: DataFrame, df_test: DataFrame = None):
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
        train(args, df_train, df_val, model, tokenizer)

    # Evaluation
    
if __name__ == "__main__":
    args = Config()
    df_train = pd.read_csv('../../data/train.csv')
    df_train = df_train.drop(columns='Unnamed: 0')
    df_val = pd.read_csv('../../data/val.csv')
    df_val = df_val.drop(columns='Unnamed: 0')
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    
    main(args, df_train, df_val)
    print("Done!")