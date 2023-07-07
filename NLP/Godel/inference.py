from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import random 
import os
import h5py
import torch.nn.functional as F
from datasketch import MinHash, MinHashLSHForest
from sklearn.feature_extraction.text import HashingVectorizer
import math
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import argparse

parser = argparse.ArgumentParser(description='Medical GODEL based chatbot')
parser.add_argument('--num', type=int,default=1,
                    help='number of relevant document return')
parser.add_argument('--thres', type=float,default=-100,
                    help='threshold for eliminating unrelevant document')
parser.add_argument('--type', type=str,default="bm25",
                    help='algorithm type of choosing relevant document')
parser.add_argument('--model_path', type=str,default="",
                    help='path to the finetune model')
parser.add_argument('--document_file', type=str,default="",
                    help='path to the document source')
parser.add_argument('--strategy', type=str,default="combine",
                    help='"combine" all the relevant document found or \
                      "best-fit" for choosing the document with highest score or\
                        "random" for choosing randomly a document in list of relevant document')

args = parser.parse_args()

root =os.getcwd()
version = "1.0.0"
if args.model_path != "":
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

tokenizer_search = AutoTokenizer.from_pretrained("roberta-base")
model_search = AutoModel.from_pretrained("roberta-base")

wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
def preprocessing(text):
   text = text.translate(str.maketrans("","",string.punctuation))
   text = text.lower().split()
   text = [wordnet_lemmatizer.lemmatize(i) for i in text if i not in nltk.corpus.stopwords.words('english')]
   return " ".join(text)
 


def bm25_score(term,doc_id, document_dict, k1 = 1.2, b=0.75):
  docCount = len(document_dict)
  docFreq = len([i for i in document_dict if term in i])
  fieldLength = len(document_dict[doc_id])
  avgFieldLength = len("".join(document_dict))/len(document_dict)
  freq = document_dict[doc_id].count(term)
  idf = math.log(1 + (docCount - docFreq + 0.5) / (docFreq + 0.5))
  tf = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (fieldLength / avgFieldLength)))
  return idf*tf

def document_search(query, document_dict, type="bm25", num=1, thres = -100, encode_doc="", save_file = "document_encoder.h5"):
  """ query: string that you want to find the relevant document
      document_dict: dictionary contain all the document you want to search for
      type: 'bm25' - based on bm25 score to find the relevant doc (elasticsearch) thres = 0.1
            'minhashLSH'- based on minhash and LSH algorithm to find relevant doc (word sequence) no thres
            'bert-cosine' - based on cosine similarity of bert embedding to find relevant doc (semantic) thres = 0.999

      num: number of relevant document at the output(default: 1)  """
  query = preprocessing(query)
  processed_doc = [preprocessing(doc) for doc in document_dict]
  if type == 'bm25':
    if thres == -100:
      thres = 0.65
    n_wordpiece = [" ".join(query.split()[i:j]) for i in  range(len(query.split())) for j in range(i+1,len(query.split())+1)]
    best = sorted([(sum([len(query)*bm25_score(query, i,processed_doc) for query in n_wordpiece])/len(n_wordpiece), document_dict[i]) for i in range(len(document_dict)) \
                   if sum([len(query)*bm25_score(query, i,processed_doc) for query in n_wordpiece])/len(n_wordpiece) > thres], reverse = True, key = lambda x: x[0])[:num]
    if len(best) == 0:
       return [(thres, "")]
    return best

  elif type == 'minhashLSH':
    vectorizer = HashingVectorizer(stop_words = 'english')
    hash = vectorizer.fit(processed_doc)
    query_embed = hash.transform([query])
    m_q = MinHash(num_perm=260)
    for i in query_embed.indices:
       m_q.update(str(i).encode("utf-8"))
    docs_embed = hash.transform(processed_doc)
    m = [MinHash(num_perm=260) for _ in docs_embed]
    lsh = MinHashLSHForest(num_perm=260)
    for id, (doc, minhash) in enumerate(zip(docs_embed,m)):
        for i in doc.indices:
            minhash.update(str(i).encode("utf-8"))
        lsh.add(id, minhash)
    lsh.index()
    result = [(m_q.jaccard(m[key]), document_dict[key])
        for key in lsh.query(m_q, 2*num) ]
    best = sorted(result, reverse = True, key = lambda x: x[0])[:num]
    if len(best) == 0:
       return [("No need thress", "")]
    return best
  elif type == 'bert-cosine':
    if thres ==-100:
      thres = 0.9996
    best = []
    query_tok = tokenizer_search("[CLS] "+query+" [SEP]", max_length=120, truncation=True, return_tensors='pt')
    query_embed = model_search(**query_tok).pooler_output
    if encode_doc =="":
      with h5py.File(save_file,"w") as hdf:
        dset = hdf.create_dataset(
            name = "document_encoder",
            shape = (len(document_dict),768),
            dtype = "float64"
        )
        for i,doc in enumerate(processed_doc):
          doc_tok = tokenizer_search("[CLS] "+doc+" [SEP]", max_length=3000, truncation=True, return_tensors='pt')
          doc_embed = model_search(**doc_tok).pooler_output
          dset[i] = doc_embed.detach().numpy()
          if F.cosine_similarity(doc_embed,query_embed).item() >thres:
            best.append((F.cosine_similarity(doc_embed,query_embed).item(),document_list[i]))
          best = sorted(best, key= lambda x:x[0], reverse=True)[:num]
    else:
      with h5py.File(encode_doc,"r") as hdf:
        for i,doc_embed in enumerate(hdf["document_encoder"]):
          if F.cosine_similarity(doc_embed,query_embed).item() >thres:
            best.append((F.cosine_similarity(doc_embed,query_embed).item(),doc))
          best = sorted(best, key= lambda x:x[0], reverse=True)[:num]
    if len(best) == 0:
       return [(thres, "")]
    return best
  else:
    raise Exception('Your method is not accepted')

def generate(knowledge, dialog):
    if knowledge != '':
        instruction = f'Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.'
        knowledge = '[KNOWLEDGE] ' + knowledge
    else:
        instruction = f'Instruction: given a dialog context, you need to response empathically.'
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

if __name__ =="__main__":
    document_list = []
    dialog = ["Hello, I am Medi, an online healcare chatbot. How can I help you today?"]
    for i in os.listdir(os.path.join(args.document_file,"chunking")):
        with open(os.path.join(args.document_file,"chunking",i),"r") as f:
            document_list.append(f.read())
    print("Medi: Hello, I am Medi, an online healcare chatbot. How can I help you today?")
    topic = ""
    temp = ""
    while True:
        
        question = input("You: ").replace("Medi","").replace("medi","")
        if question.lower() == "quit":
            break
        dialog.append(question)
        if " this " in question or " these " in question or " those " in question:
          topic = temp
        else:
          topic = ""
          temp = ""
        print("Search query:", topic+question)
        relevant_doc = document_search(topic+question, document_list,num = args.num, thres = args.thres,type= args.type)
        if args.strategy == "combine":
           knowledge = " ".join([i[1] for i in relevant_doc])
        if args.strategy == "best-fit":
           knowledge = relevant_doc[0][1]
        if args.strategy == "random":
           knowledge = random.choice(relevant_doc)[1]
        print("Document: "+ str(knowledge))
        temp = " ".join(knowledge.split()[:2]) + " "
        answer = generate(knowledge, dialog)
        print("Medi: "+ str(answer))
        dialog.append(answer)
        
