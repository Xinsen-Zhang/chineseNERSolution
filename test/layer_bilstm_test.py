# encoding:utf-8

import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../"))

from models.layers.bi_lstm import BiLSTM


if __name__ == "__main__":
    batch_input = torch.randn([128, 49, 300])
    length = torch.ones([128]).fill_(value=49).long()
    bilstm = BiLSTM(num_layers=2, hidden_size=512, input_size=300)
    batch_output = bilstm(batch_input, length)
    print(batch_output.shape)
