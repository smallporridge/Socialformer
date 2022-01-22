from util import load_json_file,multiprocess,norm
import numpy as np
from tqdm import tqdm
import argparse

'''
after finding a document word that appear in the query, we use this function to update weights of its surrounding words
'''
def update_weight(w,pos):
    '''
    Args:
        w: [list] a list of stored dynamic distance weight
        pos: [int] the index of the word
    '''
    for i in range(pos-args.delta,pos+args.delta):
        # check if the index is out of bounds
        if i>=len(w) or i<0:
            continue
        
        # the formula of dynamic distance, see the paper for more details
        w[i]+=1/(abs(pos-i)//args.window_length+1)
    return w

'''
calculate dynamic distance weight of a passage
'''
def get_dynamic_distance_weight(dic):
    '''
    Args:
        dic: a train-file-format dictionary, {'qry':[xx,xx],'psg':[xx,xx]}
    Return:
        a list stored dynamic distance weight of the qry-psg pair
    '''
    qry=dic['qry']
    psg=dic['psg']
    w=[0 for _ in range(len(psg))]
    for i in range(len(psg)):
        word=psg[i]
        if word in qry:
            update_weight(w,i)
    return list(norm(w))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta",
                        default=512,
                        type=int,
                        help="The farthest distance that query term can affect.")
    parser.add_argument("--window_length",
                        default=5,
                        type=int,
                        help="The query term will have equal effect on words in the same window.")
    args = parser.parse_args()

    # loading demo training file
    data=load_json_file(data_path="data/demo.json")

    # use multiprocess function to accelerate
    w_dynamic_list=multiprocess(get_dynamic_distance_weight,data)

    # save the w_dynamic_list
    with open("weights/dynamic.txt","w") as f:
        for i in tqdm(w_dynamic_list):
            print(i,file=f)

