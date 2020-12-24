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
        # print("DEBUG")
        # print(score.shape)
        # print(self.start_transitions.shape)
        # print(emissions[0].shape)
        # TODO 此处是否需要 logsumexp?
        for i in range(1, seq_length):
            # print("DEBUG")
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)
            # print(f"broadcast_score shape:{broadcast_score.shape}")
            # print(f"broadcast_emissions shape:{broadcast_emissions.shape}")
            # print(f"transition shape: {self.transitions.shape}")
            # print(broadcast_score+broadcast_emissions)
            next_score = self.transitions + broadcast_score
            next_score = next_score + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        score = torch.logsumexp(score, dim=1)
        return score

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """[summary]

        Args:
            emissions (torch.Tensor): [description]
            mask (Optional[torch.ByteTensor], optional): Defaults to None.
            pad_tag (Optional[int], optional): [description]. Defaults to None.

        Returns:
            List[List[List[int]]]: [description]
        """
        # TODO 检查返回的最优路径的shape
        if mask is None:
            mask = torch.ones(emissions.shape[:2], device=emissions.device,
                              detype=torch.uint8)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validation(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) \
            -> List[List[int]]:
        """使用viterbi算法解码出最优路径
            动态规划的思想

        Args:
            emissions (torch.FloatTensor): (seq_length, batch_size, num_tags)
                                            发射分数
            mask (torch.ByteTensor): (seq_length, batch_size)
                                            mask 矩阵

        Returns:
            List[List[int]]: (batch_size, seq_length)
        """
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = tags.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), fill_value=pad_tag,
                             dtype=torch.long, device=device)
        for i in range(seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions
            next_score = next_score + broadcast_emissions
            # next_score: (batch_size, self.num_tags)
            # indices: (batch_size, self.num_tags)
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i-1] = indices

        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0)-1

        # shape: (batch_size, seq_length, num_tags)
        history_idx = history_idx.transpose(0, 1).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1,
                                                               self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1,
                                                           self.num_tags))
        history_idx = history_idx.transpose(0, 1).contiguous()
        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)


if __name__ == "__main__":
    batch_size = 128
    seq_length = 64
    num_tags = 7
    crf = CRF(num_tags, batch_first=False)
    tags = torch.randint(0, 6, size=(seq_length, batch_size))
    mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
    emissions = torch.randn(seq_length, batch_size, num_tags)
    print(crf)
    score = crf._compute_score(emissions, tags, mask)
    # print(score)
    # print(score.shape)
    normalizer = crf._compute_normalizer(emissions, mask)
    # print(normalizer)
    # print(normalizer.shape)
    llh = crf(emissions, tags)
    print(llh)
    print(llh.shape)
