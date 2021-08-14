import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from transformers import XLMRobertaForSequenceClassification

"""
직접 모델을 구성하여 만들었습니다.
MyBert - model_tunning_example.py에 있는 BertForSequenceClassification튜닝과정과는 다르게 
R-Bert논문 구현 식으로 층을 추가하여 학습을 진행하였습니다.
"""

class MyBert(nn.Module):
    """
    Bert Pretrained
    """
    def __init__(self, num_labels : int = 42):
        super(MyBert,self).__init__()
        self.backbone = BertForSequenceClassification.from_pretrained(CFG.model_name, num_labels = num_labels)

        self.classifier_entity1 = nn.Linear(768, 768)
        self.classifier_entity2 = nn.Linear(768,768)
        self.classifier_cls = nn.Linear(768,768)
        self.concat_classifier = nn.Linear(768 * 3,  num_labels)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
        self.my_dropout = nn.Dropout(0.1)

        self.loss = CrossEntropyLoss()
        self.num_labels = num_labels

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ent_ids = None, # 추가 작업
        ):
        outputs, last_hidden_states, pooled_output, ent_ids_result = self.backbone(
          input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids,
          position_ids=position_ids,
          head_mask=head_mask,
          inputs_embeds=inputs_embeds,
          labels=labels,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
          ent_ids = ent_ids, # 추가 작업
        )
        if ent_ids is not None:
            batch_size = last_hidden_states.shape[0]
            token_len = last_hidden_states.shape[1]
            hidden_state_len = last_hidden_states.shape[2]

            last_hidden_state_outputs =last_hidden_states
            ent_ids_outputs = ent_ids_result
            
            entity1_output = []
            entity2_output = []
            for bs in range(batch_size):
                cut_list = []
                # 첫 처리
                if ent_ids_outputs[bs][0] == 1:
                    cut_list.append(0)

                for tl in range(1,token_len):
                    if ent_ids[bs][tl-1] == 1 and ent_ids[bs][tl] == 0:
                        cut_list.append(tl)
                    if ent_ids[bs][tl-1] == 0 and ent_ids[bs][tl] == 1:
                        cut_list.append(tl)
                
                # 끝처리
                iif len(cut_list) == 3 or len(cut_list) == 1:
                    cut_list.append(token_len)

                assert len(cut_list) % 2 == 0

                entity1_insert = torch.zeros(hidden_state_len).to(input_ids.device)
                entity2_insert = torch.zeros(hidden_state_len).to(input_ids.device)

                if len(cut_list) >= 4:
                    len1 = (cut_list[1] - cut_list[0])
                    len2 = (cut_list[3] - cut_list[2])

                    for i in range(cut_list[0], cut_list[1]):
                        entity1_insert += last_hidden_state_outputs[bs][i]
                    entity1_insert /= len1

                    for i in range(cut_list[2], cut_list[3]):
                        entity2_insert += last_hidden_state_outputs[bs][i]
                    entity2_insert /= len2
                elif len(cut_list) >= 2:
                    len1 = (cut_list[1] - cut_list[0])

                    for i in range(cut_list[0], cut_list[1]):
                        entity1_insert += last_hidden_state_outputs[bs][i]
                    entity1_insert /= len1

                entity1_insert = entity1_insert.cpu().detach()
                entity2_insert = entity2_insert.cpu().detach()
                entity1_output.append(entity1_insert.numpy())
                entity2_output.append(entity2_insert.numpy())


            entity1_output= torch.tensor(entity1_output).to(input_ids.device)
            entity2_output = torch.tensor(entity2_output).to(input_ids.device)  

            pooled_output = self.classifier_cls(pooled_output)
            entity1_output = self.classifier_entity1(self.tanh(entity1_output))
            entity2_output = self.classifier_entity2(self.tanh(entity2_output))

            logits = torch.cat([pooled_output,entity1_output,entity2_output],dim = 1)
            logits = self.my_dropout(logits)
            logits = self.concat_classifier(logits)
            logits = self.softmax(logits)

            loss_fct = self.loss # 추가 부분
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs.loss = loss
            outputs.logits = logits

        return outputs


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)