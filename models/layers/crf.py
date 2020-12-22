import torch
from torch import nn
from typing import Optional, List


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
        """
        CRF 层前向计算, 求得分
        P(y|x)=\frac{exp \sum_k w_k f_k(y_{i-1}, y_{i}, x, i)}
                    {\sum_y exp \sum_k w_k f_k(y_{i-1}, y_{i}, x, i)}
        log P(y|x) = \sum_k w_k f_k(y_{i-1}, y_i, x, i) -
                log({\sum_y exp \sum_k w_k f_k(y_{i-1}, y_{i}, x, i)})
        求出来的是极大似然分数, 优化需要-1*log likelihood
        """
        if reduction not in ["none", "sum", "mean", "token_mean"]:
            raise ValueError(
                'reduction expected  "none", "sum", "mean", "token_mean",' +
                f'but got {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validation(emissions, tags=tags, mask=mask)

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
                    tags: Optional[torch.LongTensor] = None,
                    mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError('emissions must have dimension of 3, ' +
                             f'got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and' +
                    ' tags must match, ' +
                    f'got {tuple(emissions.shape[:2])}' +
                    f' and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and' +
                    ' mask must match, ' +
                    f'got {tuple(emissions.shape[:2])}' +
                    f' and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        """ 计算batch sentences 在当前路径(tags)下的得分

        Args:
            emissions (torch.Tensor): (seq_length, batch_size, num_tags) 发射分数
            tags (torch.LongTensor): (seq_length, batch_size)  每个 token 的标签
            mask (torch.ByteTensor): (seq_length, batch_size) mask 矩阵

        Returns:
            torch.Tensor: (batch_size,)
        """
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]  # 从 start 转移到 tags[0]的分数
        # 加上从 tags[0]发射到第一个词的分数
        score = score + emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # 从 tags[i-1]转移到 tags[i]的分数
            score += self.transitions[tags[i-1], tags[i]] * mask[i]
            # 从 tags[i]发射到第 i 个单词的分数
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # 得出最后一个 tag
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        """计算 batch setences 在当前发射分数下的总得分(规范化因子)

        Args:
            emissions (torch.Tensor): (seq_length, batch_size, num_tags) 发射分数
            mask (torch.ByteTensor): (seq_length, batch_size) mask 矩阵

        Returns:
            torch.Tensor: (batch_size,)
        """
        seq_length = emissions.size()[0]
        # batch_size = emissions.size()[1]
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        # TODO 此处是否需要 logsumexp?
        for i in range(1, seq_length):
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = self.transitions + broadcast_score
            next_score = next_score + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        score = torch.logsumexp(score)
        return score

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        pass
