# encoding:utf-8
import torch
# from torch import nn

from typing import Optional, Tuple


def prepare_pack_paded_sequence(input_words: torch.FloatTensor,
                                seq_length: torch.LongTensor,
                                descending: Optional[bool] = True,
                                batch_first: Optional[bool] = True) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[summary]

    Args:
        input_words (torch.FloatTensor): [description]
        seq_length (torch.LongTensor): [description]
        descending (Optional[bool], optional): [description]. Defaults to True.
    """
    if batch_first:
        dim = 0
    else:
        dim = 1
    sorted_seq_length, indices = torch.sort(
        seq_length, dim=dim, descending=descending)
    # print(indices.shape)
    _, desorted_indices = torch.sort(
        sorted_seq_length, descending=False, dim=dim)
    # print(desorted_indices)
    sorted_input_words = input_words[indices] if batch_first else input_words[:, indices]
    return sorted_input_words, sorted_seq_length, desorted_indices


if __name__ == "__main__":
    input_words = torch.randn([3, 2, 4])
    length = torch.LongTensor([1, 3, 2]).view(3, 1)
    # print(length)
    sorted_input_words, sorted_seq_length, desorted_indices = \
        prepare_pack_paded_sequence(
            input_words, length, batch_first=True)
    print(f'input words: {input_words}')
    print(f'sequence length: {length}')
    print(f'sorted_input_words: {sorted_input_words}')
    print(f'desorted indices: {desorted_indices}')
    print(f'restored words: {sorted_input_words[desorted_indices]}')
