import os
import json
import random
def combine_file(dir):
  data=[]
  for i in os.listdir(dir):
    with open(os.path.join(dir,i), "rb") as f:
      data += json.load(f)
  with open(os.path.join(dir,"full_data.json"), "w") as k :
    json.dump(data, k, indent = 4)
def split_data(source_dir,dest_dir,test=0.1,train= 0.8,val=0.1):
  with open(source_dir,"rb") as f:
    data = json.load(f)
  assert test + train+val == 1
  data_train = random.choices(data, k = len(data)*train )
  test_val_data = [i for i in data if i not in data_train]
  data_test =random.choices(test_val_data, k = len(data)*test )
  data_val = [i for i in test_val_data if i not in data_test]
  with open(os.path.join(dest_dir,"train.json"), "w") as k :
    json.dump(data_train, k, indent = 4)
  with open(os.path.join(dest_dir,"test.json"), "w") as k :
    json.dump(data_test, k, indent = 4)
  with open(os.path.join(dest_dir,"validation.json"), "w") as k :
    json.dump(data_val, k, indent = 4)
if __name__ == "__main__":
  root =os.getcwd()
  dir = root + '/processed_data'
  if "full_data.json" not in os.listdir(dir):
    combine_file(dir)
  if "train.json" not in os.listdir(dir): 
    split_data(dir+"/full_data.json",dir)