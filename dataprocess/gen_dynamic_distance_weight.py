from util import load_json_file,multiprocess,norm
import numpy as np
from tqdm import tqdm
import argparse

def update_weight(w,pos):
    for i in range(pos-args.delta,pos+args.delta):
        if i>=len(w) or i<0:continue
        w[i]+=1/(abs(pos-i)//args.group_length+1)
    return w

def get_dynamic_distance_weight(dic):
    qry=dic['qry']
    psg=dic['psg']
    w=[0 for i in range(len(psg))]
    for i in range(len(psg)):
        word=psg[i]
        if word in qry:
            update_weight(w,i)
    return list(norm(w))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta",
                        default=512,
                        type=int)
    parser.add_argument("--group_length",
                        default=5,
                        type=int)
    args = parser.parse_args()

    data=load_json_file(root_path="data/demo.json")
    w_dynamic_list=multiprocess(get_dynamic_distance_weight,data)
    with open("weights/dynamic.txt","w") as f:
        for i in tqdm(w_dynamic_list):
            print(i,file=f)

