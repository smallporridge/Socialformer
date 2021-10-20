import json
from tqdm import tqdm
from multiprocessing import Pool
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

# normalization
def norm(array):
    ma=np.max(array)
    mi=np.min(array)
    if ma==mi:return array
    return (array-mi)/(ma-mi)

def multiprocess(func,li,pool_size=16):
    p=Pool(pool_size)        
    ans_li=p.map(func,li)  
    p.close()
    p.join()   
    return ans_li


def load_json_file(root_path):
    print("--------------------loading files---------------------")
    papers=[]
    file = open(root_path, 'r', encoding='utf-8')  
    for line in tqdm(file.readlines()):
        papers.append(json.loads(line))   
    print("-------------------------------------------------------")
    return papers


def matrix_visualize(name,matrix):
    root_path="./results/"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    save_path=root_path+"/"+name+".pdf"
    # Display matrix
    
    plt.matshow(matrix.astype(int))
    plt.tight_layout()
    plt.savefig(save_path)

