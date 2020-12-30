# encoding:utf-8
from ..layers.embedding_layer import EmbeddingLayer
from ..layers.bi_lstm import BiLSTM

from torch import nn
import torch


class BiLSTMModel(nn.Module):
    def __init__(self, model_config,
                 embedding_dim,
                 num_classes,
                 vocab_size,
                 embedding_weight):
        super(BiLSTMModel, self).__init__()
        self.batch_first = model_config['batch_first']
        self.embedding = EmbeddingLayer(vocab_size=vocab_size,
                                        embedding_weight=embedding_weight,
                                        embedding_dim=embedding_dim,
                                        p_dropout=model_config['dropout_emb'],
                                        is_frozen=model_config['is_frozen'])
        self.lstm = BiLSTM(input_size=embedding_dim,
                           hidden_size=model_config['hidden_size'],
                           num_layers=model_config['num_layers'],
                           is_bidirectional=model_config['is_bidirectional'],
                           p_dropout=model_config['dropout_lstm'],
                           num_classes=num_classes)  # bilstm的类别个数新增两个
        # self.crf = CRF(device=device, tagset_size=num_classes)

    def forward(self, inputs, masks):
        length = torch.sum(masks, dim=1).contiguous().view(-1) if self.batch_first \
            else torch.sum(masks, dim=0).contiguous().view(-1)
        x = self.embedding(inputs)
        x = self.lstm(x, length)
        return x

    def inference(self, inputs, masks):
        output_logits = self.forward(inputs, masks)
        predict_tags = torch.argmax(output_logits, dim=-1)
        return predict_tags

    def inference_and_decode(self, inputs, masks, id2tag):
        predict_tags = self.inference(self, inputs, masks).long().detach().cpu().numpy().tolist()
        batch_size = predict_tags.size()[0]
        length = torch.sum(masks, dim=1).long().detach().cpu().numpy().reshape(-1)
        result = []
        for i in range(batch_size):
            result.append([])
            seq_length = length[i]
            for j in range(seq_length):
                result[-1].append(id2tag[predict_tags[i][j]])
        return result
