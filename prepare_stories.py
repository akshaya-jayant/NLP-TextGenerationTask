import json
import re
from glob import glob
from tqdm import tqdm
import argparse
import random
import os
import itertools
import pandas as pd

def get_arguments():
    arg_parser = argparse.ArgumentParser() 
    arg_parser.add_argument('--data_size', type=int, default=10000000000)
    arg_parser.add_argument('--data_path', type=str)
    arguments = arg_parser.parse_args() 
    return arguments

arguments = get_arguments()

info=[] 
random.seed(0)

info=[]
col=["sentence"+str(i) for i in range(5)]

for fs in glob(os.path.join(arguments.data_path,"*.csv")):
    data_from_fs=pd.read_csv(fs).values
    
    for d in data_from_fs:
        info.append(" ".join(d[2:7]))

info=[{"target":d} for d in info]
info=random.sample(info, len(info))

training=info[int(len(info)*0.0):int(len(info)*0.8)]
validation=info[int(len(info)*0.8):int(len(info)*0.9)]
testing=info[int(len(info)*0.9):int(len(info)*1.0)]

print("Total data size is : ",len(info))
print("Training data size is : ",len(training))
print("Validation data size is : ",len(validation))
print("Testing data size is : ",len(testing))

with open(os.path.join(arguments.data_path, "train.json"), "w", encoding="utf-8")as f:
    json.dump(training, f, indent=4)

with open(os.path.join(arguments.data_path, "val.json"), "w", encoding="utf-8")as f:
    json.dump(validation, f, indent=4)

with open(os.path.join(arguments.data_path, "test.json"), "w", encoding="utf-8")as f:
    json.dump(testing, f, indent=4)


