import random
import torch
import datasets
from typing import Union, List, Tuple, Dict
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm

class PointDataset(Dataset):
    def __init__(self, filename, sub_graph, max_groups, max_psglen, tokenizer, dataset_script_dir, dataset_cache_dir):
        self._filename = filename
        self._tokenizer = tokenizer
        self.max_psglen = max_psglen
        self.max_groups = max_groups
        self.sub_graph = sub_graph

        self.ir_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'qry': [datasets.Value('int32')],
                'psg1': [[datasets.Value('int32')]],
                'psg2': [[datasets.Value('int32')]],
                'label': datasets.Value('int32'),
            })
        )['train']

        self.total_len = len(self.ir_dataset)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        irdata = self.ir_dataset[item]

        encoded_qry = irdata['qry']
        passages = irdata[self.sub_graph][:16+self.max_groups]
        label = irdata['label']

        if len(passages) < 16+self.max_groups:
            passages = [[] for i in range(16+self.max_groups)]

        input_ids_2d = []
        token_type_ids_2d =[]
        attention_mask_2d =[]
        passage_mask = []
        for i in range(len(passages)):
            if len(passages[i]) > 1:
                encoding = self._tokenizer.encode_plus(encoded_qry, passages[i], truncation=True, max_length=self.max_psglen + 5, padding='max_length')
                passage_mask.append(1)
            else:
                encoding = self._tokenizer.encode_plus(encoded_qry, truncation=True, max_length=self.max_psglen + 5, padding='max_length')
                passage_mask.append(0)
            input_ids_2d.append(encoding['input_ids'])
            token_type_ids_2d.append(encoding['token_type_ids'])
            attention_mask_2d.append(encoding['attention_mask'])
            
        # return encoding
        return {
            "input_ids": np.array(input_ids_2d),
            "token_type_ids": np.array(token_type_ids_2d),
            "attention_mask": np.array(attention_mask_2d),
            "passage_mask": np.array(passage_mask),
            "label": int(label)
        }