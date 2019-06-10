import torch
from torch import nn
import torch.nn.functional as F

init_pstd = 0.541324855 # ln(e-1)

class NormalizedStyleBank(nn.Module):

	def __init__(self, num_classes, code_size, class_freq):
		super(NormalizedStyleBank, self).__init__()
		self.mean = nn.Parameter(torch.zeros(num_classes, code_size))
		self.pstd = nn.Parameter(torch.Tensor(num_classes, code_size).fill_(init_pstd))
		self.register_buffer('class_freq', class_freq)

	def get_stats(self):
		full_mean = torch.matmul(self.class_freq, self.mean)
		full_ex2 = torch.matmul(self.class_freq, self.mean.pow(2) + F.softplus(self.pstd).pow(2))
		full_std = (full_ex2 - full_mean.pow(2) + 1e-8).sqrt()
		return full_mean, full_std

	def get_normalized_cov(self):
		full_mean, full_std = self.get_stats()
		mean = self.mean.sub(full_mean.unsqueeze(0)).div(full_std.unsqueeze(0))
		std = F.softplus(self.pstd).div(full_std.unsqueeze(0))
		return (torch.matmul(mean.transpose(0, 1), mean) + torch.diag(std.pow(2).sum(0))) / self.mean.size(0) 

	def forward(self, input):
		full_mean, full_std = self.get_stats()
		return torch.matmul(input, self.mean).sub(full_mean.unsqueeze(0)).div(full_std.unsqueeze(0)), torch.matmul(input, F.softplus(self.pstd)).div(full_std.unsqueeze(0))
		
	def get_code_loss(self):
		full_mean, full_std = self.get_stats()
		mean = self.mean.sub(full_mean.unsqueeze(0)).div(full_std.unsqueeze(0))
		std = F.softplus(self.pstd).div(full_std.unsqueeze(0))
		return torch.matmul(self.class_freq, (mean.pow(2) + std.pow(2)) * 0.5 - std.log() - 0.5)

class MLPClassifier(nn.Module):

	def __init__(self, in_features, hidden_features, num_layers, num_classes):
		super(MLPClassifier, self).__init__()

		net = []
		net.append(modules.Linear(in_features, hidden_features))
		net.append(modules.BatchStatistics())
		net.append(modules.CPReLU(hidden_features))
		for i in range(num_layers - 1):
			net.append(modules.LinearResidueBlock(hidden_features, residue_ratio = 1 / (i + 2), batch_reg = True))
		net.append(modules.Linear(hidden_features, num_classes))
		self.net = nn.Sequential(*net)

	def forward(self, input, mask = None):
		if mask is None:
			mask = torch.ones_like(input)
		input = input.mul(mask)
		return F.log_softmax(self.net(input), dim = 1)
