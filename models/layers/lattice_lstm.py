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
from torch.autograd import Variable
import torch.autograd as autograd


class WordLSTMCell(nn.Module):
    """ the class of lstm cell for word level(char level)

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
        """ forward calculate output

        Args:
            input_tensor (torch.Tensor): the input tensor after word embeddings
            hx (Tuple[torch.Tensor, torch.Tensor], optional): tuple of hidden state tensor and cell state tensor. Defaults to None.

        Returns:
            torch.Tensor: the cell state after forard process
        """
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
    """ the class for multi input lstm cell

        Args:
            input_size (int, optional): the input size of lstm, also the embedding dim after word embeddings. Defaults to 128.
            hidden_size (int, optional): the hidden size for hidden state and cell state of lstm cell. Defaults to 512.
        """

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

    def forward(self, input: torch.Tensor, skip_input_list: List, hidden=None):
        """
        """
        skip_input = skip_input_list[0]
        if not self.left2right:
            # TODO reverse skip_input_list
            pass
        input = input.transpose(0, 1)
        seq_length = input.size(0)
        batch_size = input.size(1)
        assert batch_size == 1
        hidden_out = []
        memory_out = []
        if hidden:
            (hx, cx) = hidden
        else:
            hx = Variable(torch.zeros(batch_size, self.hidden_dim))
            cx = Variable(torch.zeros(batch_size, self.hidden_dim))
            if self.use_gpu:
                hx = hx.cuda()
                cx = cx.cuda()

        id_list = list(range(seq_length))
        input_c_list = init_list_of_objects(seq_length)
        if not self.left2right:
            id_list = list(reversed(id_list))
        for t in id_list:  # word level index
            (hx, cx) = self.rnn(input[t], input_c_list[t], (hx, cx))
            hidden_out.append(hx)
            memory_out.append(cx)
            if skip_input[t]:
                # word_ids, words_length
                matched_num = len(skip_input[t][0])
                with torch.no_grad():
                    word_var = autograd.Variable(
                        torch.LongTensor(skip_input[t][0]))
                if self.use_gpu:
                    word_var = word_var.cuda()
                word_emb = self.word_embeddings(word_var)
                word_emb = self.word_dropout(word_emb)
                ct = self.word_rnn(word_emb, (hx, cx))
                assert ct.size(0) == len(skip_input[t][1])
                for idx in range(matched_num):
                    length = skip_input[t][1][idx]
                    if self.left2right:
                        input_c_list[t+length -
                                     1].append(ct[idx, :].unsqueeze(0))
                    else:
                        input_c_list[t-length +
                                     1].append(ct[idx, :].unsqueeze(0))
        if not self.left2right:
            hidden_out = list(reversed(hidden_out))
            memory_out = list(reversed(memory_out))
        output_hidden, output_memory = torch.cat(
            hidden_out, 0), torch.cat(memory_out, 0)
        return output_hidden.unsqueeze(0), output_memory.unsqueeze(0)


def init_list_of_objects(size: int = 10):
    """init a list of lists

    Args:
        size ([int]): the length of returned list

    Returns:
        list: a list containing size empty list
    """
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects

def convert_forward_gaz_to_backward(forward_gaz):
    length = len(forward_gaz)
    backward_gaz = init_list_of_objects(length)
    for idx in range(length):
        assert len(forward_gaz[idx]) == 2
        num = len(forward_gaz[idx][0])
        for idy in range(num):
            the_id = forward_gaz[idx][0][idy]
            the_length = forward_gaz[idx][1][idy]
            new_pos = idx + length - 1
            if backward_gaz[new_pos]:
                backward_gaz[new_pos][0].append(the_id)
                backward_gaz[new_pos][1].append(the_length)
            else:
                backward_gaz[new_pos] = [[the_id], [the_length]]
    return backward_gaz


if __name__ == "__main__":
    word_cell = WordLSTMCell()
    input_tensors = torch.randn(64, 50, 128)
    print(input_tensors[:, 0, :].shape)
    c_1 = word_cell(input_tensors[:, 0, :])
    print(c_1.shape)
