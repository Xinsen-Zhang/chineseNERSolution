# encoding:utf-8
import torch
from torch import nn
from typing import Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .layer_utils import prepare_pack_paded_sequence
# from torch.nn import functional as F
from .spatial_dropout import SpatialDropout


class BiLSTM(nn.Module):
    def __init__(self,
                 num_layers: Optional[int] = 2,
                 hidden_size: Optional[int] = 512,
                 p_dropout: Optional[float] = 0.5,
                 input_size: Optional[int] = 128,
                 is_bidirectional: Optional[bool] = True,
                 num_classes: Optional[int] = 9,
                 #  pad_index: Optional[int] = 1,
                 batch_first: Optional[bool] = True):
        """
            初始化一个 BiLSTM 层
        Args:
            num_layers (Optional[int], optional): LSTM的层数. Defaults to 2.
            hidden_size (Optional[int], optional): 隐藏层大小. Defaults to 512.
            p_dropout (Optional[float], optional): dropout 的概率. Defaults to 0.5.
            input_size (Optional[int], optional): 输入给 lstm 的 feature size. Defaults to 128.
            is_bidirectional (Optional[bool], optional): 是否使用双向 LSTM. Defaults to True.
            num_classes (Optional[int], optional): 全连接层的logits 数量. Defaults to 9.
            batch_first (Optional[bool], optional): Defaults to True.
        """
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout
        self.is_bidirectional = is_bidirectional
        self.directional_num = 2 if self.is_bidirectional else 1
        self.batch_first = batch_first
        self.dropout = SpatialDropout(self.p_dropout)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=p_dropout, bidirectional=is_bidirectional)

        self.linear = nn.Linear(in_features=hidden_size*self.directional_num,
                                out_features=num_classes)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, inputs: torch.FloatTensor, length: torch.LongTensor) \
            -> torch.Tensor:
        sorted_inputs, sorted_seq_length, desorted_indices = \
            prepare_pack_paded_sequence(
                inputs, length, batch_first=self.batch_first)
        embedding_packed = pack_padded_sequence(
            sorted_inputs, sorted_seq_length.cpu(), batch_first=self.batch_first)
        output, _ = self.lstm(embedding_packed)
        output, _ = pad_packed_sequence(output, batch_first=self.batch_first)
        output = output[desorted_indices] if self.batch_first \
            else output[:, desorted_indices]
        output = self.dropout(output)
        output = torch.tanh(output)
        logits = self.linear(output)
        return logits


if __name__ == "__main__":
    batch_input = torch.randn([128, 49, 300])
    length = torch.ones([128]).fill_(value=49).long()
    bilstm = BiLSTM(num_layers=2, hidden_size=512, input_size=300)
    batch_output = bilstm(batch_input, length)
    print(batch_output.shape)
