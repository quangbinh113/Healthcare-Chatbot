# GODEL

`chunk_data.py` : chunk data into small part. Each part contain 7 sentences. Any chunk <200 character had been deleted

```
python chunk_data.py
```

`data_gen_free.py` : generate data by chatgpt with the context save in chunking folder. We use [aecheong08/ChatGPT](`https://github.com/acheong08/ChatGPT`) to gen data via web UI. I give you 4 access_key in `key.env` expired in 4/7/2023.

```
python data_gen_free.py --save_file 
```

`data_gen_paid.py` : generate data by chatgpt with the context save in chunking folder. We use official OpenAI API for gen data. No key for you.

```
python data_gen_paid.py --save_file 
```

`data_transformation.py` : transformation data to train our two model. 

* `type` :
  * "dia": conversation_data -> dialog_knowledge_data
  * "conv": dialog_knowledge_data -> conversation_data

```
python data_gen_paid.py --save_file --filepath --type
```

`split_data.py` : combine all json data in the same folder to one and split into train, test and validation set

```
python split_data.py --source_dir --dest_dir 
```

`train.py` : train godel model and evaluate it

* `train:` path to train file
* `test`: path to test file
* `dev`: path to validaiton file
* `save_path`: where to save model
* `length`: max length of tokenizer

```
python train.py \
  --train \
  --test \
  --dev \
  --save_path \
  --length
```

`inference.py` : train godel model and evaluate it

* `model_path:` path to the model if "" will take default model of godel
* `num`: number of documents return
* `type`: type of retrive context, I propose 3 methods: `bert-cosine`, `minhashLSH` and `bm25`
* `strategy`: way to use document, I propose 3 kinds: `combine` (combine `num` document found in a string), `random` (random choose a document in `num` document) and `best` ( choose the document have best score)
* `thres`: threshold to remove unrelated document

```
python inference.py \
  --model_path \
  --num \
  --type \
  --strategy \
  --thres
```
