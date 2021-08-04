from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import Dropout, Linear, CrossEntropyLoss


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,config, num_labels = 3):
        super().__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self,input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output,pooler_output,hidden = self.bert(input_ids, token_type_ids, attention_mask).to_tuple()
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
            return loss
        else:
            return logits