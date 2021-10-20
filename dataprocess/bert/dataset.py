
import datasets
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm

'''
    This function is used to calculate the cosine similarity between qry_embeddings and psg_embeddings
'''
def get_att_dis(qry_embeddings, psg_embeddings,top_k,padding):
    '''
    Args:
        qry_embeddings([torch.tensor])
        psg_embeddings([torch.tensor)
        top_k([int]): number of psg_embedding to pick up
        padding([int]): aligned length of the attention_distribution
    '''
    # use the average of qry_embeddings to represent query
    qry_embedding=torch.sum(qry_embeddings,dim=0)/qry_embeddings.shape[0]

    # calculate the cosine similarity between qry_embedding and psg_embeddings
    attention_distribution = [torch.cosine_similarity(qry_embedding, psg_embedding,dim=0) 
                                            for psg_embedding in psg_embeddings]
    attention_distribution = torch.Tensor(attention_distribution)

    # pick up the indices of top_k largest attention weight
    if len(psg_embeddings)>top_k:
        _ ,indices=attention_distribution.topk(top_k,largest=True,sorted=True)
    else:
        indices=[i for i in range(len(psg_embeddings))]
        indices.extend([0 for i in range(len(indices),top_k)])
    indices=torch.tensor(np.array(indices))

    # padding
    delta=padding-attention_distribution.shape[0]
    pad = nn.ZeroPad2d(padding=(0, delta))  
    attention_distribution=pad(attention_distribution)
  
    return attention_distribution,indices


class TrainDataset(Dataset):
    def __init__(self,device,top_k, bert_model, filename, tokenizer, dataset_script_dir, dataset_cache_dir):
        self._filename = filename
        self._max_seq_length = top_k
        self._tokenizer = tokenizer
        self._bert_model=bert_model
        self._device=device
        self.top_k=top_k

        self.ir_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'qry': [datasets.Value("int32")],
                'psg': [datasets.Value("int32")],
                'label': datasets.Value("int32"),
            })
        )['train']

        self.total_len = len(self.ir_dataset)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        irdata = self.ir_dataset[item]

        qry_tensor=torch.LongTensor(np.array(irdata['qry'])).to(self._device)
        psg_tensor=torch.LongTensor(np.array(irdata['psg'])).to(self._device)
        
        qry_embeddings=self._bert_model.embeddings.word_embeddings(qry_tensor)
        psg_embeddings=self._bert_model.embeddings.word_embeddings(psg_tensor)

        # calculate cos_weight and the corresponding top_k indices
        cos_weight,indices=get_att_dis(qry_embeddings,psg_embeddings,top_k=self.top_k,padding=2048)

        # only the top_k indices will be put into bert, others use cos_weight
        encoded_qry = irdata['qry']
        encoded_psg = [irdata['psg'][i] for i in indices]

        encoding = self._tokenizer.encode_plus(
            encoded_qry,
            encoded_psg,
            truncation='only_second',
            max_length=self._max_seq_length,
            padding='max_length',
        )

        return {
            "input_ids": np.array(encoding['input_ids']),
            "token_type_ids": np.array(encoding['token_type_ids']),
            "attention_mask": np.array(encoding['attention_mask']),
            "label": int(irdata['label']),
            'qry_length':int(len(irdata['qry'])),
            'psg_length':int(len(irdata['psg'])),
            'cos_weight':cos_weight,
            'indices':indices
        }