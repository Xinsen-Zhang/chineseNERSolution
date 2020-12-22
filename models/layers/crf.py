import torch
from torch import nn


class CRF(nn.Module):

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        """初始化 CRF 层

        Args:
            num_tags (int): tags 的数量(不算 start 和 end)
            batch_first (bool, optional): batch_first. Defaults to False.

        Raises:
            ValueError: if num_tags <= 0
        """
        if num_tags <= 0:
            raise ValueError(
                f"invalid number of tags: {num_tags}, " +
                "it should be greater than 0")
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(self.num_tags))
        # shape:(num_tags)
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags))
        # shape: (num_tags)
        self.transitions = nn.Parameter(
            torch.empty(self.num_tags, self.num_tags))
        # shape: (num_tags, num_tags)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
