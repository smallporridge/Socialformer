
from collections import defaultdict
import math
from tqdm import tqdm
from util import load_json_file,norm
import numpy as np

'''
calculate tfidf based probability matrix, corresponding to the Static Centrality in paper
'''
def get_tfidf_weight(passages):
    '''
    Args:
        passages: a list of passages, each passage is a list of token_ids
    Return:
        a dictionary: {key:token_id, value: tfidf value}
    '''
    doc_frequency=defaultdict(int)
    for list in passages:
        for word in list:
            doc_frequency[word]+=1
 
    # calculate tf value
    tf={}  
    for word in doc_frequency:
        tf[word]=doc_frequency[word]/sum(doc_frequency.values())
 
    # calculate idf value
    doc_num=len(passages)
    idf={} 
    doc=defaultdict(int) # doc number including the word
    for word in doc_frequency:
        for psg in passages:
            if word in psg:
                doc[word]+=1
    for word in doc_frequency:
        idf[word]=math.log(doc_num/(doc[word]+1))
 
    # calculate tf_idf value
    tf_idf={}
    for word in doc_frequency:
        tf_idf[word]=tf[word]*idf[word]
 
    return tf_idf

if __name__=="__main__":
    data=load_json_file(data_path="./data/demo.json")
    psg_list=[dic['psg'] for dic in data]

    with open("./weights/tfidf.txt","w") as f:
        tf_idf=get_tfidf_weight(psg_list)
        # precalculate tfidf weight list of each passage
        for i in tqdm(range(len(psg_list))):
            w=[]
            for j in range(len(psg_list[i])):
                word=psg_list[i][j]
                w.append(tf_idf[word])
            print(list(norm(w)),file=f)


