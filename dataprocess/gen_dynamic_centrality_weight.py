import torch
from torch.utils.data import DataLoader
from transformers import BertModel
from transformers import BertTokenizer
from bert.model import BertForCLSWeight
from bert.dataset import TrainDataset
from tqdm import tqdm
import argparse
import numpy as np
from util import norm
import os

'''
Due to the time limitation of bert, we use a two-stage process.

First calculate cosine similarity of the query words and each document word,
use cosine similarity as temporary document weight.

Then we select words with the top_k biggest weight, and put them into bert to get attention weight as their final weight.
'''
def get_attention_weight():

    # load pretrained model
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    bert_model = BertModel.from_pretrained(args.model_path,output_attentions=True)
    model=BertForCLSWeight(bert_model)
    model=model.to(device)

    model.eval()
    test_dataset = TrainDataset(device,90,bert_model,args.passage_file_path,tokenizer, args.dataset_script_dir, args.dataset_cache_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    
    cls_attention_list = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, leave=False)    
        for _ , dic in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in dic.keys():
                    dic[key] = dic[key].to(device)

            # get CLS_attention_weight       
            attention_weight_batch = model(dic)
            attention_weight_batch = attention_weight_batch.cpu().numpy()  

            # cosine similarity of the query words and each document word
            dic['cos_weight'] = dic['cos_weight'].cpu().numpy()

            # the indices of words with the top_k biggest weight
            dic['indices'] = dic['indices'].cpu().numpy()

            # replace top_k cos weight with corresponding attention weight        
            for t in range(dic['cos_weight'].shape[0]):             
                cos_weight = torch.Tensor(dic['cos_weight'][t])
                indices = torch.round(torch.Tensor(dic['indices'][t])).int()
                cos_topk = torch.index_select(cos_weight,dim=0,index = indices).numpy()

                # Align attention_weight with cos_weight 
                attention_weight_batch[t] = attention_weight_batch[t]*np.sum(cos_topk)/np.sum(attention_weight_batch[t])

                # Combine attention_weight with cos_weight by indices
                cnt = 0  
                for i in dic['indices'][t]:    
                    if cnt >= attention_weight_batch[t].shape[0]:break  
                    dic['cos_weight'][t][i] = attention_weight_batch[t][cnt]
                    cnt += 1

                # normalize and save
                cls_attention_list.append(list(norm(dic['cos_weight'][t])))

    return cls_attention_list


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_gpu_test_batch_size",
                        default=4,
                        type=int,
                        help="The batch size per gpu.")
    parser.add_argument("--model_path",
                        default="./bert/model",
                        type=str)
    parser.add_argument("--dataset_script_dir",
                        default="./bert/data_scripts",
                        type=str)
    parser.add_argument("--dataset_cache_dir",
                        default="./bert/dataset_cache",
                        type=str)
    parser.add_argument("--passage_file_path",
                        default="data/demo.json",
                        type=str,
                        help="Path of dataset, each line of dataset is a dictionary like{'qry':[xx,xx],'psg':[xx,xx],'label':0/1}")
    parser.add_argument("--cpu", 
                        default=False,
                        type=bool)
    args = parser.parse_args()

    # get absolute path
    root=os.getcwd()
    args.passage_file_path=os.path.join(root,args.passage_file_path)

    if args.cpu:
        device = torch.device("cpu")
        args.test_batch_size = args.per_gpu_test_batch_size * 1

    else:
        device = torch.device("cuda:0")
        args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()

    torch.multiprocessing.set_start_method('spawn')
    read_path="./data/demo.json"
    write_path="./weights/bert.txt"

    # calculate attention weight and save
    w_attention_list=get_attention_weight()
    with open(write_path,"w") as f:
        for i in tqdm(w_attention_list):
            print(i,file=f)

