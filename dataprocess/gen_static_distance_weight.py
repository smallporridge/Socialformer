from util import norm,multiprocess
import numpy as np
import pickle
from tqdm import tqdm


'''
calculate static distance based probability matrix of size n*n
'''
def get_static_distance_matrix(n):
    '''
    Args: 
        n: [int] node number
    Return:
        normalized probability matrix
    '''
    probability_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            '''
            P_{sd}(i,j)=\frac{1}{(1+|i-j|/p)^2}
            '''
            probability_matrix[i][j]=probability_matrix[j][i]=1/((j-i)//50+1)**2
    return norm(probability_matrix)

'''
load the precalculate static distance based probability matrix   
'''
def load_static(node_num,root_path="./weights/static/"):
    path=root_path+str(node_num)+".txt"
    with open(path,"rb") as f:
        matrix=pickle.load(f)
    return np.array(matrix)


if __name__=="__main__":
    # precalculate static distance based probability
    for i in tqdm(range(1,2050)):
        w=get_static_distance_matrix(i)
        path="./weights/static/"+str(i)+".txt"
        with open(path,"wb") as f:
            pickle.dump(w,f)


