import json
from tqdm import tqdm
from multiprocessing import Pool
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os


'''
Normalization
'''
def norm(array):
    max_value=np.max(array)
    min_value=np.min(array)
    if max_value == min_value:
        return array
    else:
        return (array-min_value)/(max_value-min_value)

'''
Use multiple processes to accelerate
'''
def multiprocess(func,args_list,pool_size=16):
    '''
    Args:
        func: the function to run
        args_list: a list, every item in the list is a set of arguments for func
        pool_size: number of processes
    '''
    p=Pool(pool_size)        
    ans_list=p.map(func,args_list)  
    p.close()
    p.join()   
    return ans_list

'''
Load training data, each line of data_path is a dictionary like: {'qry':[xx,xx], 'psg':[xx,xx]}
'''
def load_json_file(data_path):
    print("--------------------loading files---------------------")
    dic_list=[]
    file = open(data_path, 'r', encoding='utf-8')  
    for line in tqdm(file.readlines()):
        dic_list.append(json.loads(line))   
    print("-------------------------------------------------------")
    return dic_list

'''
Visualize a numpy matrix
'''
def matrix_visualize(save_path,matrix):
    '''
    Args:
        save_path: str
        matrix: numpy.ndarray
    '''
    plt.matshow(matrix.astype(int))
    plt.tight_layout()
    plt.savefig(save_path)

