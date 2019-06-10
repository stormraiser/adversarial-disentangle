import math
import torch
from torch import nn
import torch.nn.functional as F

class CPReLU(nn.Module):

	def __init__(self, num_features = 1, init = 0.25):
		super(CPReLU, self).__init__()
		self.num_features = num_features
		self.weight = nn.Parameter(torch.Tensor(num_features).fill_(init))

		self.post_mean = (1 - init) / math.sqrt(2 * math.pi)
		post_ex2 = (1 + init ** 2) / 2
		self.post_std = math.sqrt(post_ex2 - self.post_mean ** 2)

	def forward(self, input):
		return (F.prelu(input, self.weight) - self.post_mean) / self.post_std
