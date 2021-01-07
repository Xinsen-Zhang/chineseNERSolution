# encoding:utf-8

import torch
from torch import nn
from typing import Tuple


class WordLSTMCell(nn.Module):
    def __init__(self, input_size=128, hidden_size=512):
        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.FloatTensor(input_size, 3*hidden_size))
        self.weight_bias = nn.Parameter(
            torch.FloatTensor(hidden_size, 3*hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3*hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.weight.data)
        nn.init.eye_(self.weight_bias)
        nn.init.zeros_(self.bias)

    def _init_hidden_and_cell_state(self, batch_size):
        h_0 = torch.randn(batch_size, self.hidden_size)
        c_0 = torch.randn(batch_size, self.hidden_size)
        return h_0, c_0

    def forward(self, input_tensor: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None):
        batch_size = input_tensor.size()[0]
        if hx is None:
            hx = self._init_hidden_and_cell_state(batch_size)
        h_0, c_0 = hx
        bias_batch = self.bias.unsqueeze(0).expand(
            batch_size, *self.bias.size())
        bias = torch.addmm(bias_batch, h_0, self.weight_bias)
        weight = torch.mm(input_tensor, self.weight)
        f, i, c = torch.split(weight+bias, self.hidden_size, 1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(c)
        return c_1


class MultiInputLSTMCell(nn.Module):
    def __init__(self, input_size=128, hidden_size=512):
        super(MultiInputLSTMCell, self).__init__()


if __name__ == "__main__":
    word_cell = WordLSTMCell()
    input_tensors = torch.randn(64, 50, 128)
    print(input_tensors[:, 0, :].shape)
    c_1 = word_cell(input_tensors[:, 0, :])
    print(c_1.shape)
