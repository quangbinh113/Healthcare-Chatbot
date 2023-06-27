import json
import argparse

parser = argparse.ArgumentParser(description='Transformation my dataset to group dataset type')
parser.add_argument('--filepath', type=str,default="medical_train.json",
                    help='transformation file path ')
parser.add_argument('--save_file', type=str,default="transformed_medical_train.json",
                    help='transformation saving file path ')
parser.add_argument('--type', type=str,default="conv",
                    help='transformation saving file path ')

args = parser.parse_args()

if __name__ == "__main__":
  filepath = args.filepath
  with open(filepath,"rb") as f:
    data =json.load(f)
  if args.type == "conv":
    transformed_train = []
    for da in data:
        data_point = {}
        for i, context in enumerate(da['dialog']):
            data_point["context" +str(i)] =context
        data_point['response'] = da['response']
        transformed_train.append(data_point)
  elif args.type == "dia":
    transformed_train = []
    for da in data:
      try:
        transformed_train.append({"dialog": list(da.values())[:-1] , "knowledge": "", "response": da["response"]})
      except Exception:
        continue
  with open(args.save_file,"w") as f:
    print(len(transformed_train))
    json.dump(transformed_train,f)
 
    