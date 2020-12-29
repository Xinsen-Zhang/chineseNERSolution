# encoding utf-8
import torch
from torch import nn
from itertools import repeat


class SpatialDropout(nn.Module):
    def __init__(self, p_dropout):
        super(SpatialDropout, self).__init__()
        self.p_dropout = p_dropout

    def forward(self, inputs: torch.FloatTensor):
        if (not self.training) or self.p_dropout == 0.0:
            return input
        noise = self._make_noise(inputs)
        return torch.mul(inputs, noise)

    def _make_noise(self, inputs: torch.FloatTensor) -> torch.Tensor:
        """
        产生一个noise, 来自input这个tensor
            相同的type, 相同的device

        Args:
            input (torch.FloatTensor): 输入的张量
        Returns:
            torch.Tensor: 产生的noise张量
        """
        noise = inputs.new()
        noise.resize_(inputs.size()[0], *repeat(1,
                                                inputs.dim()-2),
                      inputs.size()[-1])
        if self.p_dropout == 1:
            return noise.fill_(0).expand_as(inputs)
        noise.bernoulli_(1-self.p_dropout).div_(1-self.p_dropout)
        return noise.expand_as(inputs)


if __name__ == "__main__":
    spatial_dropout = SpatialDropout(0.2)
    batch_tensor = torch.randn(size=(3, 4, 5))
    out_tensor = spatial_dropout(batch_tensor)
    print(out_tensor)
    print(out_tensor.shape)
