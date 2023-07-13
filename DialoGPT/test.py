import evaluate 
from config import Config 
from utils import  load_and_cache_examples
from typing import List, Dict
import torch
from pandas import DataFrame 
from transformers import  PreTrainedModel, PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,SequentialSampler
from tqdm.notebook import tqdm, trange
import os
import pandas as pd

def evaluate(args: Config, 
             model: PreTrainedModel, 
             tokenizer: PreTrainedTokenizer, 
             df_test: DataFrame, 
             prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, df)
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last = True
    )
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    
    try:
        with open(os.path.join(eval_output_dir, "eval_results.txt"), "w") as writer:
            writer.write(f"perplexity: {perplexity}\n loss: {eval_loss}")
    except Exception as e:
        print(e)

    result = {"perplexity": perplexity, "loss": eval_loss}
    
    return result

if __name__ == "__main__":
    args = Config()
    model = None
    tokenizer = None
    path= "data/DialoGPT_format/csv_data/test.csv"
    df_test = pd.read_csv(path)
    evaluate(args, model, tokenizer, df_test)
    print("Done!")