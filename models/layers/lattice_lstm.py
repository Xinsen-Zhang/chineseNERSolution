#! /usr/bin/bash python
# -*- encoding: utf-8 -*-
'''
@filename		: models/layers/lattice_lstm.py
@description	:
@created_time	: 2021/01/11 23:18:41
@author		: xinsen
version		: 1.0
'''

import torch
from torch import nn
from typing import Tuple, List


class WordLSTMCell(nn.Module):
    """[summary]

        Args:
            input_size (int, optional): the size of input tensors(embedding dims after word_embeddings). Defaults to 128.
            hidden_size (int, optional): the size of hidden size(after lstm forward, the dim for cell state and hidden state). Defaults to 512.
        """
    def __init__(self, input_size=128, hidden_size=512):
        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.FloatTensor(input_size, 3*hidden_size))
        self.weight_hidden = nn.Parameter(
            torch.FloatTensor(hidden_size, 3*hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3*hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.weight.data)
        nn.init.eye_(self.weight_hidden)
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
        bias = torch.addmm(bias_batch, h_0, self.weight_hidden)
        weight = torch.mm(input_tensor, self.weight)
        f, i, c = torch.split(weight+bias, self.hidden_size, 1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(c)
        return c_1


class MultiInputLSTMCell(nn.Module):
    def __init__(self, input_size=128, hidden_size=512):
        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.FloatTensor(input_size, 3*hidden_size))
        self.weight_hidden = nn.Parameter(
            torch.FloatTensor(hidden_size, 3*hidden_size))
        self.alpha_weight = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))
        self.alpha_hidden = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size))
        self.alpha_bias = nn.Parameter(hidden_size)
        self.bias = nn.Parameter(torch.FloatTensor(3*hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.orthogonal_(self.alpha_weight)
        nn.init.zeros_(self.weight_hidden)
        nn.init.zeros_(self.alpha_hidden)
        nn.init.zeros_(self.alpha_bias)

    def _init_hidden_and_cell_state(self, batch_size):
        h_0 = torch.randn(batch_size, self.hidden_size)
        c_0 = torch.randn(batch_size, self.hidden_size)
        return h_0, c_0

    def forward(self, input_tensor: torch.Tensor, c_inputs: List(torch.Tensor), hx: Tuple(torch.FloatTensor, torch.FloatTensor) = None):
        """多源输入 LSTM

        Args:
            input_tensor (torch.Tensor): input 张量
            c_inputs (List): a list of tensors with length c_nums, each element is the input c_t from skip word(batch, hidden_size)
            hx (Tuple, optional): [description]. Defaults to None.
        """
        batch_size = input_tensor.size()[0]
        if hx is None:
            hx = self._init_hidden_and_cell_state(batch_size)
        h_0, c_0 = hx
        assert batch_size == 1
        # (batch_size, hidden_size)
        bias_batch = self.bias.unsqueeze(0).expand(
            batch_size, *self.bias.size())
        weight_bias = torch.addmm(bias_batch, h_0, self.weight_hidden)
        weight = torch.mm(input_tensor, self.weight)
        weight = weight_bias + weight
        input_gate, output_gate, memory_cell = torch.split(
            weight, self.hidden_size, 1)
        input_gate = torch.sigmoid(input_gate)
        output_gate = torch.sigmoid(output_gate)
        memory_cell = torch.tanh(memory_cell)
        candidate_num = len(c_inputs)
        if candidate_num == 0:
            c_1 = (1-input_gate)*c_0 + input_gate*memory_cell
            h_1 = output_gate * torch.tanh(c_1)
        else:
            # (batch_size*c_num, hidden_dim)
            c_input_tensor = torch.cat(c_inputs, 0)
            # double check
            # (c_num, hidden_dim)
            c_input_tensor = c_input_tensor.squeeze(c_input_tensor)
            alpha_bias = torch.addmm(self.alpha_bias.unsqueeze(0).expand(
                batch_size, self.hidden_size), input_tensor, self.alpha_weight).expand(candidate_num, self.hidden_size)
            alpha_weight = torch.mm(c_input_tensor, self.alpha_hidden)
            alpha_weight = alpha_bias + alpha_weight
            alpha = torch.sigmoid(alpha_weight)
            # (1+c_num, hidden_dim)
            alpha = torch.exp(torch.cat([input_gate, alpha], 0))
            temp = torch.sum(alpha, 0)
            alpha = torch.div(alpha, temp)
            temp = torch.cat([memory_cell, c_input_tensor], 0)
            c_1 = temp * alpha
            h_1 = output_gate * torch.tanh(c_1)
        return h_1, c_1


class LatticeLSTM(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512,
                 left2right: bool = False, word_p_dropout: float = 0.5,
                 word_alphabet_size: int = 10240,
                 word_embedding_dim: int = 256, pretrained_work_embeddings=None,
                 fix_word_embeddings: bool = False, use_gpu: bool = True):
        super(LatticeLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.word_alphabet_size = word_alphabet_size
        self.word_embedding_dim = word_embedding_dim
        self.word_embeddings = nn.Parameter(
            word_alphabet_size, word_embedding_dim)
        if pretrained_work_embeddings is not None:
            self.word_embeddings.data.copy_(
                torch.from_numpy(pretrained_work_embeddings))
        else:
            self.word_embeddings.data.copy_(
                torch.from_numpy(self.random_embeddings())
            )
        self.fix_word_embeddings = fix_word_embeddings
        if fix_word_embeddings:
            self.word_embeddings.requires_grad = False  # freeze grad
        self.p_word_dropout = word_p_dropout
        self.word_dropout = nn.Dropout(word_p_dropout)
        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim)
        self.word_rnn = WordLSTMCell(word_embedding_dim, hidden_dim)
        self.left2right = left2right
        # skip_direction = "forward" if left2right else "backward"
        self._reset_parameters()
        if self.use_gpu:
            self.rnn = self.rnn.cuda()
            self.word_dropout = self.word_dropout.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.word_rnn = self.word_rnn.cuda()

    def _reset_parameters(self):
        pass

    def random_embeddings(self):
        return torch.randn((self.word_alphabet_size, self.word_embedding_dim))

    def forward(self, input, skip_input_list, hidden=None):
        """
        """


if __name__ == "__main__":
    word_cell = WordLSTMCell()
    input_tensors = torch.randn(64, 50, 128)
    print(input_tensors[:, 0, :].shape)
    c_1 = word_cell(input_tensors[:, 0, :])
    print(c_1.shape)
