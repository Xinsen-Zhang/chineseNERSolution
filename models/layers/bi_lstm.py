# encoding:utf-8
import torch
from torch import nn
from typing import Optional


class BiLSTM(nn.Module):
    def __init__(self,
                 num_layers: Optional[int] = 2,
                 hidden_size: Optional[int] = 512,
                 p_dropout: Optional[float] = 0.5,
                 input_size: Optional[int] = 128,
                 is_bidirectional: Optional[bool] = True,
                 num_classes: Optional[int] = 9,
                 pad_index: Optional[int] = 1):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout
        self.is_bidirectional = is_bidirectional
        self.directional_num = 2 if self.is_bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=p_dropout, bidirectional=is_bidirectional)

        self.linear = nn.Linear(in_features=hidden_size*self.directional_num,
                                out_features=num_classes)
        nn.init.xavier_normal(self.linear.weight)

    def forward(self, inputs: torch.FloatTensor, length: torch.LongTensor) \
            -> torch.Tensor:
        pass
