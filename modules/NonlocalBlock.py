import torch
from torch import nn
import torch.nn.functional as F

from .NromInitModules import Conv2d, Linear
from .BatchRegularization import BatchRegularization

def nonlocal_read_write(write_value, write_key, read_key):
	batch_size = write_value.size(0)
	value_channels = write_value.size(1)
	key_channels = write_key.size(1)
	spatial_size = write_value.size()[2:]
	spatial_num = write_value.numel() // batch_size // value_channels

	write_key = write_key.contiguous().view(batch_size, key_channels, spatial_num)
	read_key = read_key.contiguous().view(batch_size, key_channels, spatial_num)
	write_value = write_value.contiguous().view(batch_size, value_channels, spatial_num)

	weight = torch.bmm(read_key.transpose(1, 2), write_key)
	weight = F.softplus(weight)
	weight = weight.div(weight.sum(2, keepdim = True).add(1e-4))
	return torch.bmm(write_value, weight.transpose(1, 2)).contiguous().view(batch_size, value_channels, *spatial_size)

class NonlocalBlock(nn.Module):

	def __init__(self, in_channels, value_channels, key_channels, num_msg = 1, residue_ratio = 0, batch_reg = False):
		super(NonlocalBlock, self).__init__()

		self.residue_ratio = nn.Parameter(torch.Tensor(in_channels).fill_(residue_ratio ** 0.5))
		self.shortcut_ratio = nn.Parameter(torch.Tensor(in_channels).fill_((1 - residue_ratio) ** 0.5))

		self.value_channels = value_channels
		self.key_channels = key_channels
		self.num_msg = num_msg

		self.write_value_map = Conv2d(in_channels, value_channels * num_msg, 1)
		self.write_key_map = Conv2d(in_channels, key_channels * num_msg, 1)
		self.read_key_map = Conv2d(in_channels, key_channels * num_msg, 1)
		self.read_value_map = Conv2d(value_channels * num_msg, in_channels, 1)

		self.batch_reg = BatchRegularization(in_channels) if batch_reg else nn.Sequential()

	def forward(self, input):
		spatial_size = input.size()[2:]
		write_value = self.write_value_map(input).contiguous().view(input.size(0) * self.num_msg, self.value_channels, *spatial_size)
		write_key = self.write_key_map(input).view(input.size(0) * self.num_msg, self.key_channels, *spatial_size)
		read_key = self.read_key_map(input).view(input.size(0) * self.num_msg, self.key_channels, *spatial_size)
		read_value = nonlocal_read_write(write_value, write_key, read_key)
		read_value = self.read_value_map(read_value.contiguous().view(input.size(0), self.value_channels * self.num_msg, *spatial_size))

		extra_dims = (1,) * (input.dim() - 2)
		return self.batch_reg(input.mul(self.shortcut_ratio.view(1, input.size(1), *extra_dims)) + read_value.mul(self.residue_ratio.view(1, input.size(1), *extra_dims)))
