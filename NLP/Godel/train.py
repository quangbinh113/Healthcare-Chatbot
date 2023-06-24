import numpy as np
import os
import nltk
from datasets import load_dataset, load_metric
import accelerate
import evaluate
import torch
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,TrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

def tokenize_function(examples):

    instruction_k = f'Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.'
    instruction_nk = f'Instruction: given a dialog context, you need to response empathically.'
    inputs = [f"{instruction_k} [CONTEXT] {' EOS '.join(dialog)} [KNOWLEDGE] {knowledge}" if knowledge != "" else\
              f"{instruction_nk} [CONTEXT] {' EOS '.join(dialog)}" for dialog, knowledge in zip(examples["dialog"], examples["knowledge"])]
    targets = [ex for ex in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=320, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target= targets, max_length=320, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0].argmax(axis = -1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [["\n".join(nltk.sent_tokenize(label.strip()))] for label in decoded_labels]

    # Metric
    #rouge
    result = metric2.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #bleu
    result1 = metric1.compute(predictions=decoded_preds, references=decoded_labels)
    result["bleu"] = result1["score"]

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


if __name__  == "__main__":
    root = os.getcwd()
    dataset = load_dataset("json", data_files={"train":"train.json", "validation":"validation.json","test":"test.json"})
    metric1 = load_metric("sacrebleu")
    metric2 = evaluate.load("rouge")
    #metric3 = meteor
    #metric4 = perplexity
    #metric5 = f1
    #metric6 = CiDr
    #metric7 = Hits@1
    #metric8 = Consistency score

    tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size = 512)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(output_dir="content",
                                  report_to="tensorboard",
                                  load_best_model_at_end = True,
                                  save_strategy="epoch",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=512,
                                  per_device_eval_batch_size=512,
                                  dataloader_num_workers=2,
                                  fp16=True,
                                  save_total_limit=1,
                                  logging_strategy="epoch",
                                  predict_with_generate=True,
                                  num_train_epochs=20,
                                  learning_rate=2e-5,
                                  weight_decay=1e-3)
    trainer =  Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(root + "/Godel/official_model/model-v1.0.0")