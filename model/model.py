from transformers import BertModel, PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.init as init

class BertForSearch(PreTrainedModel):
    def __init__(self, bert_model, max_docs, max_groups, max_psglen):
        super(BertForSearch, self).__init__(bert_model.config)
        self.embeddings = bert_model.embeddings
        self.layer = bert_model.encoder.layer
        self.pooler = bert_model.pooler
        self.config = bert_model.config
        self.passage_transformer = nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=512, batch_first=True)
        #self.classifier = nn.Linear(768, 2)
        self.score = nn.Linear(768, 1)
        self.mlp = nn.Linear(768*2, 768)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.max_docs = max_docs
        self.max_groups = max_groups
        self.max_psglen = max_psglen
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        init.xavier_normal_(self.score.weight)
    
    def forward(self, batch_data, pooling='max', structure='hierarchical',is_training=True):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        input_ids = batch_data["input_ids"] #[bsz, max_docs, max_groups, max_psglen]
        
        attention_mask = batch_data["attention_mask"]
        token_type_ids = batch_data["token_type_ids"]
        passage_mask = batch_data["passage_mask"]
        passage_mask = 1-passage_mask
        target_labels = batch_data["label"]

        input_ids = input_ids.reshape(-1, self.max_psglen)
        attention_mask = attention_mask.reshape(-1, self.max_psglen)
        token_type_ids = token_type_ids.reshape(-1, self.max_psglen)
        passage_mask = passage_mask.reshape(-1, self.max_groups)

        input_shape = input_ids.size()
        device = input_ids.device    

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)    

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        ) 

        if structure == "hierarchical":
            for i, layer_module in enumerate(self.layer):    

                layer_head_mask = head_mask[i] if head_mask is not None else None    

                
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                )    

                hidden_states = layer_outputs[0] #[bsz*grz, sql, 768]
                leader_states = hidden_states[:,0]    

                leader_states = leader_states.reshape(-1, self.max_groups, 768)
                leader_states_after = self.passage_transformer(leader_states, src_key_padding_mask=passage_mask.bool())
                leader_states = self.mlp(torch.cat([leader_states, leader_states_after], 2))
                leader_states = leader_states.reshape(-1, 768)   

                hidden_states[:,0] = leader_states    

        if structure == "plain":
            for i, layer_module in enumerate(self.layer):    

                layer_head_mask = head_mask[i] if head_mask is not None else None    

                
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                    layer_head_mask,
                )    

                hidden_states = layer_outputs[0] #[bsz*grz, sql, 768]  

        pooled_output = self.pooler(hidden_states)
        pooled_output = pooled_output.reshape(-1, self.max_groups, 768)

        if pooling == 'max':
            pooled_output = (passage_mask * -10000.0).unsqueeze(-1) + pooled_output
            pooled_output = torch.max(pooled_output, 1)[0]

        scores = self.score(self.dropout(pooled_output))

        if is_training:
            scores = scores.reshape(-1, self.max_docs)
            loss = self.cross_entropy(scores, target_labels)
            return loss
        else:
            scores = scores.squeeze()
            return scores
