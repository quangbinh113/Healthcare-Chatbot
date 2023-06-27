from datasets.features.features import pa
import numpy as np
import os
import nltk
import argparse
from datasets import load_dataset
import accelerate
import evaluate
import torch
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,TrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")


parser = argparse.ArgumentParser(description='Transformation my dataset to group dataset type')
parser.add_argument('--train', type=str,default="",
                    help='train file path ')
parser.add_argument('--dev', type=str,default="",
                    help='validation file path ')
parser.add_argument('--test', type=str,default="",
                    help='test file path ')
parser.add_argument('--save_path', type=str,default="",
                    help='save path for model')
parser.add_argument('--length', type=int,default=320,
                    help='length of tokenizer')

args = parser.parse_args()

def tokenize_function(examples):

    instruction_k = f'Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.'
    instruction_nk = f'Instruction: given a dialog context, you need to response empathically.'
    inputs = [f"{instruction_k} [CONTEXT] {' EOS '.join(dialog)} [KNOWLEDGE] {knowledge}" if knowledge != "" else\
              f"{instruction_nk} [CONTEXT] {' EOS '.join(dialog)}" for dialog, knowledge in zip(examples["dialog"], examples["knowledge"])]
    targets = [ex for ex in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=args.length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target= targets, max_length=args.length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class f1:
  def compute(self,predictions, references, type = 'marco'):
    f1s =[]
    precisions = []
    recalls = []
    for i in range(len(predictions)):
      precision = 0
      recall = 0
      for j in " ".join(predictions[i]).split():
        if j in " ".join(references[i][0]).split():
          precision += 1
      for j in " ".join(references[i][0]).split():
        if j in " ".join(predictions[i]).split():
          recall += 1
      print((len(" ".join(predictions[i]).split())+1),len(" ".join(references[i][0]).split()))
      p = precision/(len(" ".join(predictions[i]).split())+1)
      r = recall/(len(" ".join(references[i][0]).split())+1)
      e = (1e-5)/(len(" ".join(predictions[i]).split())+len(" ".join(references[i][0]).split())+2)
      precisions.append(p)
      recalls.append(r)
      f1s.append(2*p*r*(p+r)/((p+r)**2 +e**2))
    if type == 'micro':
      return {'f1': sum(f1s)/len(f1s)}
    if type == 'marco':
      e_a = (1e-5)/(len(precisions)+len(recalls))
      p_a = sum(precisions)/len(precisions)
      r_a = sum(recalls)/len(recalls)
      return {'f1': 2*p_a*r_a*(p_a+r_a)/((p_a+r_a)**2 +e_a**2)}



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
    result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #bleu
    result1 = metric1.compute(predictions=decoded_preds, references=decoded_labels)
    result2["bleu"] = result1["bleu"]
    #meteor
    result3 = metric3.compute(predictions=decoded_preds, references=decoded_labels)
    result2["meteor"] = result3["meteor"]
    #perplexity
    result4 = metric4.compute(predictions=decoded_preds, model_id='gpt2')
    result2["perplexity"] = result4['mean_perplexity']
    #f1
    result5 = metric5.compute(predictions=decoded_preds, references=decoded_labels)
    result2['f1'] = result5['f1']

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result2["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result2.items()}



if __name__  == "__main__":
    root = os.getcwd()
    dataset = load_dataset("json", data_files={"train": args.train, "validation":args.dev,"test":args.test})
    metric1 = evaluate.load("bleu")
    metric2 = evaluate.load("rouge")
    metric3 = evaluate.load("meteor")
    metric4 = evaluate.load("perplexity", module_type="metric")
    metric5 = f1()

    tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size = 32)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(output_dir="content",
                                  report_to="tensorboard",
                                  load_best_model_at_end = True,
                                  save_strategy="epoch",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=128,
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
    trainer.save_model(args.save_path+"/model-v1.0.0")
    trainer.evaluate(tokenized_datasets["test"])