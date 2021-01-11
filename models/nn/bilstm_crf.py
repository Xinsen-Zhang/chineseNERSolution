#! /usr/bin/bash python
# -*- encoding: utf-8 -*-
'''
@filename		: models/nn/bilstm_crf.py
@description	:
@created_time	: 2021/01/11 23:22:07
@author		: xinsen
version		: 1.0
'''


from ..layers.embedding_layer import EmbeddingLayer
from ..layers.bi_lstm import BiLSTM
from ..layers.crf import CRF

from torch import nn
import torch


class BiLSTMCRFModel(nn.Module):
    def __init__(self, model_config,
                 embedding_dim,
                 num_classes,
                 vocab_size,
                 embedding_weight):
        super(BiLSTMCRFModel, self).__init__()
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
        self.crf = CRF(num_classes, self.batch_first)
        self.device = model_config['device']

    def forward(self, inputs, masks, tags=None):
        length = torch.sum(masks, dim=1).contiguous().view(-1) if self.batch_first \
            else torch.sum(masks, dim=0).contiguous().view(-1)
        x = self.embedding(inputs)
        x = self.lstm(x, length)
        mask = masks.byte()
        if tags is None:
            result = self.crf.decode(x, mask)
            seq_length = inputs.size(
            )[1] if self.batch_first else inputs.size()[1]
            # TODO 不是 batch_first 的时候怎么进行 Implement?
            if self.batch_first:
                for item in result:
                    if len(item) < seq_length:
                        res_num = seq_length - len(item)
                        for i in range(res_num):
                            item.append(0)
            result = torch.LongTensor(result).to(self.device)
            return result
        else:
            return self.crf(x, tags, mask=mask, reduction='token_mean')

    def inference(self, inputs, masks):
        predict_tags = self.forward(inputs, masks)
        # print(predict_tags.shape)
        return predict_tags

    def inference_and_decode(self, inputs, masks, id2tag):
        predict_tags = self.inference(
            self, inputs, masks).long().detach().cpu().numpy().tolist()
        batch_size = predict_tags.size()[0]
        length = torch.sum(masks, dim=1).long(
        ).detach().cpu().numpy().reshape(-1)
        result = []
        for i in range(batch_size):
            result.append([])
            seq_length = length[i]
            for j in range(seq_length):
                result[-1].append(id2tag[predict_tags[i][j]])
        return result
