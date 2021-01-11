#! /usr/bin/bash python
# -*- encoding: utf-8 -*-
'''
@filename		: models/layers/embedding_layer.py
@description	:
@created_time	: 2021/01/11 23:20:43
@author		: xinsen
version		: 1.0
'''

import torch
from torch import nn
from .spatial_dropout import SpatialDropout
from typing import Optional
import numpy as np


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 embedding_weight: Optional[np.ndarray] = None,
                 is_frozen: Optional[bool] = False,
                 p_dropout: Optional[float] = 0.25):
        """ embedding 层对象

        Args:
            embedding_dim (int): 嵌入维数
            vocab_size (int): 词表大小(数量)
            embedding_weight (Optional[np.ndarray], optional): 
                            Defaults to None. 是否指定权重, None的话则随机初始化
            is_frozen (Optional[bool], optional):  Defaults to False.
                                                是否冻结嵌入层参数
            p_dropout (Optional[float], optional): Defaults to 0.25.
                                                dropout的概率
        """
        super(EmbeddingLayer, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.is_frozen = is_frozen
        if self.is_frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if embedding_weight is not None:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_weight))
        else:
            self.encoder.weight.data.copy_(torch.from_numpy(
                self._generate_random_embeddings(vocab_size, embedding_dim)))

        self.p_dropout = p_dropout
        self.dropout = SpatialDropout(p_dropout)

    def _generate_random_embeddings(self, vocab_size: int,
                                    embedding_dim: int) -> np.ndarray:
        embeded_matrix = np.empty([vocab_size, embedding_dim])
        for i in range(vocab_size):
            embeded_matrix[i, :] = np.random.uniform(-1, 1, [1, embedding_dim])
        return embeded_matrix

    def forward(self, inputs: torch.LongTensor) -> torch.Tensor:
        x = self.encoder(inputs)  # (batch_size, seq_length, embedding_dim)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    embedding_layer = EmbeddingLayer(718, 10240, is_frozen=False).to(device)
    input_tensor = torch.randint(0, 718, [128, 56]).to(device)
    output_tensor = embedding_layer(input_tensor)
    print(output_tensor.shape)
