import torch
from torch import nn
import torch.nn.functional as F

from .NromInitModules import Conv2d, Linear
from .BatchRegularization import BatchRegularization

def pool(value, weight):
	batch_size = value.size(0)
	bus_channels = value.size(1)
	spatial_num = value.numel() // batch_size // bus_channels

	value = value.contiguous().view(batch_size, bus_channels, spatial_num)
	weight = weight.contiguous().view(batch_size, bus_channels, spatial_num)
	weight = F.softplus(weight)
	weight = weight.div(weight.sum(2, keepdim = True).add(1e-4))

	return torch.mul(value, weight).sum(2)

class BusBlock(nn.Module):

	def __init__(self, in_channels, bus_channels, residue_ratio = 0, batch_reg = False):
		super(BusBlock, self).__init__()

		self.residue_ratio = nn.Parameter(torch.Tensor(in_channels).fill_(residue_ratio ** 0.5))
		self.shortcut_ratio = nn.Parameter(torch.Tensor(in_channels).fill_((1 - residue_ratio) ** 0.5))

		self.write_value_map = Conv2d(in_channels, bus_channels, 1)
		self.weight_map = Conv2d(in_channels, bus_channels, 1)
		self.read_value_map = Linear(bus_channels, in_channels)

		self.batch_reg = BatchRegularization(in_channels) if batch_reg else nn.Sequential()

	def forward(self, input):
		write_value = self.write_value_map(input)
		weight = self.weight_map(input)
		extra_dims = (1,) * (input.dim() - 2)

		pool_value = self.read_value_map(pool(write_value, weight))
		return self.batch_reg(input.mul(self.shortcut_ratio.view(1, input.size(1), *extra_dims)) + pool_value.mul(self.residue_ratio.unsqueeze(0)).view(input.size(0), input.size(1), *extra_dims))
