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

def get_bert_weight():

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
            bert_weight_batch = model(dic)
            bert_weight_batch=bert_weight_batch.cpu().numpy()   
            dic['cos_weight']=dic['cos_weight'].cpu().numpy()
            dic['indices']=dic['indices'].cpu().numpy()
                    
            for t in range(dic['cos_weight'].shape[0]):             
                cos_weight=torch.Tensor(dic['cos_weight'][t])
                indices=torch.round(torch.Tensor(dic['indices'][t])).int()
                cos_topk=torch.index_select(cos_weight,dim=0,index=indices).numpy()

                # Align bert_weight with cos_weight 
                bert_weight_batch[t]=bert_weight_batch[t]*np.sum(cos_topk)/np.sum(bert_weight_batch[t])

                # Combine bert_weight with cos_weight by indices
                cnt=0  
                for i in dic['indices'][t]:    
                    if cnt>=bert_weight_batch[t].shape[0]:break  
                    dic['cos_weight'][t][i]=bert_weight_batch[t][cnt]
                    cnt+=1

                cls_attention_list.append(list(norm(dic['cos_weight'][t])))

    return cls_attention_list


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod",
                        default="train_2048",
                        type=str)
    parser.add_argument("--per_gpu_test_batch_size",
                        default=4,
                        type=int,
                        help="The batch size.")
    parser.add_argument("--model_path",
                        default="bert/model",
                        type=str)
    parser.add_argument("--dataset_script_dir",
                        default="bert/data_scripts",
                        type=str)
    parser.add_argument("--dataset_cache_dir",
                        default="bert/dataset_cache",
                        type=str)
    parser.add_argument("--passage_file_path",
                        default="data/demo.json",
                        type=str)
    parser.add_argument("--cpu", 
                        default=True,
                        type=bool)
    parser.add_argument("--delta", 
                        default=0,
                        type=int)
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
        args.test_batch_size = args.per_gpu_test_batch_size * 1

    else:
        device = torch.device("cuda:0")
        args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()

    torch.multiprocessing.set_start_method('spawn')
    read_path="./data/demo.json"
    write_path="./weights/bert.txt"

    w_bert_list=get_bert_weight()
    with open(write_path,"w") as f:
        for i in tqdm(w_bert_list):
            print(i,file=f)

