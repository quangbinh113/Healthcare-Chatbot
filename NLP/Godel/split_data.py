import os
import json
import random
import argparse

parser = argparse.ArgumentParser(description='Transformation my dataset to group dataset type')
parser.add_argument('--source_dir', type=str,default="",
                    help='file to transform ')
parser.add_argument('--dest_dir', type=str,default="",
                    help='file transform to ')
args = parser.parse_args()

def combine_file(source_dir, dest_dir):
  data=[]
  for i in os.listdir(source_dir):
    try:
      with open(os.path.join(source_dir,i), "rb") as f:
        data += json.load(f)
    except Exception:
      continue
  with open(os.path.join(dest_dir,"full_data.json"), "w") as k :
    json.dump(data, k, indent = 4)
def split_data(source_dir,dest_dir,test=0.1,train= 0.8,val=0.1):
  with open(source_dir,"rb") as f:
    data = json.load(f)
  assert test + train+val == 1
  print("Total_data: ",len(data))
  data_train = list(random.sample(data, k = int(len(data)*train) ))
  test_val_data = [i for i in data if i not in data_train]
  data_test =list(random.sample(test_val_data, k = int(len(data)*test) ))
  data_val = [i for i in test_val_data if i not in data_test]
  with open(os.path.join(dest_dir,"train.json"), "w") as k :
    json.dump(data_train, k, indent = 4)
  with open(os.path.join(dest_dir,"test.json"), "w") as k :
    json.dump(data_test, k, indent = 4)
  with open(os.path.join(dest_dir,"validation.json"), "w") as k :
    json.dump(data_val, k, indent = 4)
  print("Train: ", len(data_train))
  print("Validation: ",len(data_val))
  print("Test: ", len(data_test))
if __name__ == "__main__":
  source_dir = args.source_dir
  dest_dir = args.dest_dir
  if "full_data.json" not in os.listdir(dest_dir):
    combine_file(source_dir,dest_dir)
  if "train.json" not in os.listdir(dest_dir): 
    split_data(dest_dir+"/full_data.json",dest_dir)