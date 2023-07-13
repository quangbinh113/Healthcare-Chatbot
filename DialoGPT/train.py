from config import Config
import torch
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
from utils import set_seed, load_and_cache_examples, save_model, read_file
from tqdm.notebook import tqdm, trange
from pandas import DataFrame
import wandb 

from transformers import AdamW, PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup


def train(args: Config,
          model: PreTrainedModel, 
          tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Wandb init """
    wandb.init(
    # set the wandb project where this run will be logged
        project="health-care chatbot",
        name=args.experiment_name,
)
    """Read data"""
    
    df_train, df_val = read_file(args.data_path)
    
    """ Train the model """

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_dataset = load_and_cache_examples(args, tokenizer, df_train)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last = True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model.resize_token_embeddings(len(tokenizer))
    # add_special_tokens_(model, tokenizer)


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Batch size = {args.train_batch_size}", )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {t_total}")

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print(f"  Continuing training from epoch {epochs_trained}")
            print(f"  Continuing training from global step {global_step}")
            print(f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")
        except ValueError:
            print("  Starting fine-tuning.")

    best_perplexity = 100000
    best_epoch = 0
    
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for i in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        total_train_loss = 0
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch, batch)
            if inputs.shape[1] > 1024: continue
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss.backward()

            avg_loss = loss.mean().item()
            total_train_loss += loss.mean().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0:
                # Log metrics
                logging_loss = loss.mean().item()
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
            wandb.log({"train_step_loss": avg_loss})
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        # eval phase 
        val_perplexity = evaluate(args, model, tokenizer, df_val)['perplexity'].item()
        val_loss = evaluate(args, model, tokenizer, df_val)['loss']
        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity 
            best_epoch = i
            # Save model checkpoint
            save_model(args, model, tokenizer, optimizer, scheduler)
        epoch = i + 1
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_perplexity = torch.exp(torch.tensor(avg_train_loss))
        wandb.log({"train_epoch_loss": avg_train_loss, 
                   "val_epoch_loss": val_loss,
                   "train_perplexity": train_perplexity, 
                   "val_perplexity": val_perplexity})
        print(f"Epoch {epoch}/{args.num_train_epochs} train loss: {avg_train_loss:.4f}, val loss: {val_loss:.4f},val perplexity: {val_perplexity:.4f}")
    wandb.finish()
    print(f"Best epoch: {best_epoch + i}, best perplexity: {best_perplexity:.4f}")   

# Evaluation of some model

def evaluate(args: Config, 
             model: PreTrainedModel, 
             tokenizer: PreTrainedTokenizer, 
             df: DataFrame, 
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