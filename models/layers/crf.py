import torch
from torch import nn
from typing import Optional


class CRF(nn.Module):

    def __init__(self, num_tags: int,
                 batch_first: Optional[bool] = False) -> None:
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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = "mean") -> torch.Tensor:
        if reduction not in ["none", "sum", "mean", "token_mean"]:
            raise ValueError(
                'reduction expected  "none", "sum", "mean", "token_mean",' +
                f'but got {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validation(emissions, tags, mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            # shape: (seq_length, batch_size, num_tags)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator  # log likelihood

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum()/mask.float().sum()  # token level mean

    def _validation(self, emissions: torch.Tensor,
                    tags: torch.LongTensor,
                    mask: Optional[torch.ByteTensor] = None) -> None:
        pass

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        pass

    def _compute_normalizer(emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        pass
