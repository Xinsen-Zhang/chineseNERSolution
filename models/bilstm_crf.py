# encoding utf-8

from .layers.crf_naive import CRF
from .layers.spatial_dropout import HighEfficiencySpatialDropout
from torch import nn


class BiLSTMCRFNERModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 label2id, device, p_dropout=0.2, num_layers=2):
        super(BiLSTMCRFNERModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self.device = device
        self.label2id = label2id
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(input_size=self.embedding_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True, dropout=self.p_dropout,
                              bidirectional=True)
        self.dropout = HighEfficiencySpatialDropout(self.p_dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size*2)
        self.classification = nn.Linear(self.hidden_size*2, len(label2id))
        self.crf = CRF(len(label2id), self.label2id, device=self.device)

    def forward(self, input_ids, input_mask):
        embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        sequence_out, _ = self.bilstm(embs)
        sequence_out = self.layer_norm(sequence_out)
        features = self.classification(sequence_out)
        return features
        # TODO 如此操作, 存疑

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            # TODO 此处存疑
            return features, self.crf.calculate_loss(features, input_tags, input_lens)
        else:
            return features
