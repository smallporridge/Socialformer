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

class ListDataset(Dataset):
    def __init__(self, filename_global, sub_graph, max_groups, max_psglen, tokenizer, dataset_script_dir, dataset_cache_dir):
        self._filename_global = filename_global
        self._tokenizer = tokenizer
        self.max_psglen = max_psglen
        self.max_groups = max_groups
        self.sub_graph = sub_graph

        self.ir_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=self._filename_global,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                'qry': {
                    'qid': datasets.Value('string'),
                    'query': [datasets.Value('int32')],
                },
                'pos': [{
                    'ori_slice': [[datasets.Value('int32')]],
                    'sub_graph1': [[datasets.Value('int32')]],
                    'sub_graph2': [[datasets.Value('int32')]],
                    'pid': datasets.Value('string'),
                }],
                'neg': [{
                    'ori_slice': [[datasets.Value('int32')]],
                    'sub_graph1': [[datasets.Value('int32')]],
                    'sub_graph2': [[datasets.Value('int32')]],
                    'pid': datasets.Value('string'),
                }]}
            )
        )['train']

        self.total_len = len(self.ir_dataset)  
      
    def __len__(self):
        return self.total_len

    def create_passage(self, encoded_qry, passages):
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
        return input_ids_2d, token_type_ids_2d, attention_mask_2d, passage_mask
    
    def __getitem__(self, item):
        irdata = self.ir_dataset[item]

        encoded_qry = irdata['qry']['query']
        pos_psg = irdata['pos']
        neg_psg = irdata['neg']

        input_ids_3d = []
        token_type_ids_3d =[]
        attention_mask_3d =[]
        passage_mask_3d = []

        for pos_entry in pos_psg:
            pos_passage = pos_entry['ori_slice'][:16] + pos_entry[self.sub_graph][:self.max_groups]
            input_ids_2d, token_type_ids_2d, attention_mask_2d, passage_mask = self.create_passage(encoded_qry, pos_passage)
            input_ids_3d.append(input_ids_2d)
            token_type_ids_3d.append(token_type_ids_2d)
            attention_mask_3d.append(attention_mask_2d)
            passage_mask_3d.append(passage_mask)

        for neg_entry in neg_psg:
            neg_passage = neg_entry['ori_slice'][:16] + neg_entry[self.sub_graph][:self.max_groups]
            input_ids_2d, token_type_ids_2d, attention_mask_2d, passage_mask = self.create_passage(encoded_qry, neg_passage)
            input_ids_3d.append(input_ids_2d)
            token_type_ids_3d.append(token_type_ids_2d)
            attention_mask_3d.append(attention_mask_2d)
            passage_mask_3d.append(passage_mask)

        # return encoding
        return {
            "input_ids": np.array(input_ids_3d),
            "token_type_ids": np.array(token_type_ids_3d),
            "attention_mask": np.array(attention_mask_3d),
            "passage_mask": np.array(passage_mask_3d),
            "label": np.array(0)
        }