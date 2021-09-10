from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads
# MLM + CLS Score BERT预训练模型
class BertForPretrain(nn.Module):
    def __init__(self, bert_model):
        super(BertForPretrain, self).__init__()
        self.bert_model = bert_model # bert encoder
        self.cls = BertPreTrainingHeads(self.bert_model.config) # 用于直接算MLM的logits, 以及CLS的hidden logits
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) # 用于算loss
        
    def forward(self, batch_data):
        input_ids = batch_data["input_ids"] # [batch_size, seq_length]
        attention_mask = batch_data["attention_mask"] # [batch_size, sequence_length]
        token_type_ids = batch_data["token_type_ids"] # [batch_size, sequence_length]
        mlm_labels = batch_data["mlm_labels"] # [batch_size, sequence_length]. 不预测的地方填-100, 不参与loss计算.
        seq_label = batch_data['label'] # [batch_size]
        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        # 过bert encoder
        sequence_output, pooled_output = self.bert_model(**bert_inputs)[:2]  # sequence_output: [batch_size, sequence_length, hidden_size]; pooled_output: [batch_size, hidden_size]
        # 过MLP, 取MLM Logitsy与CLS logits
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output) # prediction_scores: [batch_size, sequence_length, vocab_size]; seq_relationship_score: [batch_size, 2] 
        # 算loss
        masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.bert_model.config.vocab_size), mlm_labels.view(-1)) # [batch_size]
        seq_rel_loss = self.loss_fct(seq_relationship_score.view(-1, 2), seq_label.view(-1)) # [batch_size]        
        total_loss = masked_lm_loss + seq_rel_loss # [batch_size]
        return total_loss
