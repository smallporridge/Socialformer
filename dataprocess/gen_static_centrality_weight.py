
from collections import defaultdict
import math
from tqdm import tqdm
from util import load_json_file,norm
import numpy as np

def get_tfidf_weight(passages):
    doc_frequency=defaultdict(int)
    for list in passages:
        for i in list:
            doc_frequency[i]+=1
 
    # calculate tf value
    tf={}  
    for i in doc_frequency:
        tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 
    # calculate idf value
    doc_num=len(passages)
    idf={} 
    doc=defaultdict(int) # doc number including the word
    for i in doc_frequency:
        for j in passages:
            if i in j:
                doc[i]+=1
    for i in doc_frequency:
        idf[i]=math.log(doc_num/(doc[i]+1))
 
    # calculate tf_idf value
    tf_idf={}
    for i in doc_frequency:
        tf_idf[i]=tf[i]*idf[i]
 
    return tf_idf

if __name__=="__main__":
    data=load_json_file(root_path="./data/demo.json")
    psg_list=[dic['psg'] for dic in data]

    with open("./weights/tfidf.txt","w") as f:
        tf_idf=get_tfidf_weight(psg_list)
        # get tfidf weight list of each passage
        for i in tqdm(range(len(psg_list))):
            w=[]
            for j in range(len(psg_list[i])):
                word=psg_list[i][j]
                w.append(tf_idf[word])
            print(list(norm(w)),file=f)


