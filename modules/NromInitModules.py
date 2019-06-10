import math
import torch
from torch import nn

eps = 1e-8

class Conv2d(nn.Conv2d):

	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = True):
		super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias)
		weight_norm = self.weight.pow(2).sum(3, keepdim = True).sum(2, keepdim = True).sum(1, keepdim = True).add(eps).sqrt()
		self.weight.data.div_(weight_norm)

class ConvTranspose2d(nn.ConvTranspose2d):

	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = True):
		super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 0, 1, bias, 1)
		norm_scale = math.sqrt(self.stride[0] * self.stride[1])
		weight_norm = self.weight.pow(2).sum(3, keepdim = True).sum(2, keepdim = True).sum(0, keepdim = True).add(eps).sqrt()
		self.weight.data.div_(weight_norm).mul_(norm_scale)

class Linear(nn.Linear):

	def __init__(self, in_features, out_features, bias = True):
		super(Linear, self).__init__(in_features, out_features, bias)
		weight_norm = self.weight.pow(2).sum(1, keepdim = True).add(eps).sqrt()
		self.weight.data.div_(weight_norm)
