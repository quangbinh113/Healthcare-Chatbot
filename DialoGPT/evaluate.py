from config import Config
from utils import load_and_cache_examples 
from transformers import PreTrainedModel, PreTrainedTokenizer 
from pandas import DataFrame 
from typing import Dict, List 
import os 
import torch 
from dataset import ConversationDataset 
from torch.utils.data import DataLoader, SequentialSampler, pad_sequence 
import tqdm 
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead
import argparse

def run(args: Config, df: DataFrame) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
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
    parser = argparse.ArgumentParser()
                        
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default="DialoGPT-small")
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--train_path', type=str, default="../data/DialoGPT_format/csv_data/train.csv")
    parser.add_argument('--val_path', type=str, default="../data/DialoGPT_format/csv_data/validation.csv")
    
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
    if parse.model.startswith("DialoGPT"):
        args.model_name_or_path = 'microsoft/' + parse.model
        args.config_name = 'microsoft/' + parse.model
        args.tokenizer_name = 'microsoft/' + parse.model
    else :
        args.model_name_or_path = parse.model
        args.config_name = parse.model
        args.tokenizer_name = parse.model
    
    run(args)
    print("Done!")