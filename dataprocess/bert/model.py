import torch.nn as nn
import torch

'''
    return attention weights(CLS layer) corresponding to the document part
'''
class BertForCLSWeight(nn.Module):
    def __init__(self,bert_model):
        super(BertForCLSWeight,self).__init__()
        self.bert_model = bert_model
    
    def forward(self,batch_data):
        '''
        Args:
            input_ids([type]):[description]
            attention_mask([type]):[description]
            token_type_ids([type]):[description]
        '''

        input_ids_,token_type_ids_,attention_mask_=batch_data['input_ids'],batch_data['token_type_ids'],batch_data['attention_mask']
        qry_len,psg_len=batch_data['qry_length'][0].item(),batch_data['psg_length'][0].item()
        bert_output=self.bert_model(input_ids=input_ids_,token_type_ids=token_type_ids_,attention_mask=attention_mask_)
                
        attentions = bert_output[2][-1] # attentions: (batch_size, num_heads, sequence_length, sequence_length) 
        attentions = torch.sum(attentions,dim=1) # get the weight sum of all heads
        attentions = attentions[:,0,1+qry_len+1:1+qry_len+1+psg_len] # extract the attention weight of passage part

        return attentions




