from util import norm,multiprocess
import numpy as np
import pickle
from tqdm import tqdm

# get static probability matrix of length n
def get_static_distance_matrix(n):
    w=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            w[i][j]=w[j][i]=1/((j-i)//50+1)**2
    return norm(w)

# load the static probability matrix    
def load_static(n,root_path="./weights/static/"):
    path=root_path+str(n)+".txt"
    with open(path,"rb") as f:
        ans=pickle.load(f)
    return np.array(ans)


if __name__=="__main__":
    for i in tqdm(range(1,2050)):
        w=get_static_distance_matrix(i)
        path="./weights/static/"+str(i)+".txt"
        with open(path,"wb") as f:
            pickle.dump(w,f)


