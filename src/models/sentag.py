import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertConfig,
    BertPreTrainedModel,
)
from .crf import CRF
from .layers import SelfAttention


class SenTag(BertPreTrainedModel):
    def __init__(self, config, label_list, device):
        super(SenTag, self).__init__(config)

        # get bert model
        self.bert = BertModel(config)
        self.config = config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.sentence_lstm = nn.LSTM(input_size=self.config.hidden_size * 2,
                                     hidden_size=self.config.hidden_size // 2,
                                     batch_first=True,
                                     bidirectional=True)

        self.self_attention = SelfAttention(hidden_size=self.config.hidden_size, dim=self.config.hidden_size)
        self.ln = LayerNorm(self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size, len(label_list))
        label2id = {k: i for i, k in enumerate(label_list)}
        self.crf = CRF(num_tags=len(label_list))

        self.init_weights()

    def forward(self, sentences_input_ids=None, sentences_input_mask=None,
                sentences_type_ids=None, sentences_input_len=None, label_ids=None):

        batch_size = sentences_input_ids.shape[0]
        seq_len = sentences_input_ids.shape[1]
        sentences_feature = list()
        for i in range(batch_size):
            bert_output = self.bert(sentences_input_ids[i], sentences_type_ids[i], sentences_input_mask[i])

            last_hidden_state = bert_output['last_hidden_state']
            pooler_output = bert_output['pooler_output']

            # we define the sentence features as the average the sequence output and concatenate with pooler output
            # mask the padding tokens for each sentence
            sentence_feature = torch.unsqueeze(sentences_input_mask[i], 2) * last_hidden_state

            # perform self attention layer to re-estimate sentence features
            sentence_feature, attention_weights = self.self_attention(sentence_feature, sentence_feature, sentence_feature)

            # compute the average pooling the sentence feature
            sentence_feature = torch.sum(sentence_feature, dim=1) / seq_len

            sentence_feature = torch.cat((sentence_feature, pooler_output), dim=1)
            sentences_feature.append(sentence_feature)

        sentences_feature = torch.stack(sentences_feature)

        # passing sentence feature to lstm
        outputs, _ = self.sentence_lstm(sentences_feature)
        outputs = self.ln(outputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        loss = None
        if label_ids is not None:
            loss = -self.crf(logits, label_ids)
        return {'loss': loss, 'logits': logits}