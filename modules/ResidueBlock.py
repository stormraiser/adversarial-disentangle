import torch
from torch import nn
import torch.nn.functional as F

from .NromInitModules import Conv2d, ConvTranspose2d, Linear
from .CPReLU import CPReLU
from .BatchRegularization import BatchRegularization

class Interpolate2x(nn.Module):

	def __init__(self, pad_h, pad_w):
		super(Interpolate2x, self).__init__()
		self.pad_h = pad_h
		self.pad_w = pad_w

	def forward(self, input):
		output = F.interpolate(input, scale_factor = 2, mode = 'nearest')
		return output[:, :, self.pad_h : output.size(2) - self.pad_h, self.pad_w : output.size(3) - self.pad_w]

class ResidueBlock(nn.Module):

	def __init__(self, in_channels, out_channels, stride, pad_h, pad_w, residue_ratio = 0, batch_reg = False, nonlinear = True):
		super(ResidueBlock, self).__init__()

		self.residue_ratio = nn.Parameter(torch.Tensor(out_channels).fill_(residue_ratio ** 0.5))
		self.shortcut_ratio = nn.Parameter(torch.Tensor(out_channels).fill_((1 - residue_ratio) ** 0.5))

		self.residue = nn.Sequential(
			Conv2d(in_channels, out_channels, stride * 3, stride, (stride + pad_h, stride + pad_w)) if stride > 0 else ConvTranspose2d(in_channels, out_channels, 6, 2, (2 + pad_h, 2 + pad_w)),
			CPReLU(out_channels) if nonlinear else nn.Sequential()
		)

		self.shortcut = nn.Sequential(
			nn.AvgPool2d(2, padding = (pad_h, pad_w)) if stride == 2 else nn.Sequential(),
			Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Sequential(),
			Interpolate2x(pad_h, pad_w) if stride == -2 else nn.Sequential()
		)

		self.batch_reg = BatchRegularization(out_channels) if batch_reg else nn.Sequential()

	def forward(self, input):
		view_shape = (1, self.residue_ratio.size(0), 1, 1)
		return self.batch_reg(self.shortcut(input).mul(self.shortcut_ratio.view(*view_shape)) + self.residue(input).mul(self.residue_ratio.view(*view_shape)))

class LinearResidueBlock(nn.Module):

	def __init__(self, num_features, residue_ratio = 0, batch_reg = False):
		super(LinearResidueBlock, self).__init__()

		self.residue_ratio = nn.Parameter(torch.Tensor(num_features).fill_(residue_ratio ** 0.5))
		self.shortcut_ratio = nn.Parameter(torch.Tensor(num_features).fill_((1 - residue_ratio) ** 0.5))

		self.residue = nn.Sequential(
			Linear(num_features, num_features),
			CPReLU(num_features)
		)

		self.batch_reg = BatchRegularization(num_features) if batch_reg else nn.Sequential()

	def forward(self, input):
		return self.batch_reg(input.mul(self.shortcut_ratio.unsqueeze(0)) + self.residue(input).mul(self.residue_ratio.unsqueeze(0)))
