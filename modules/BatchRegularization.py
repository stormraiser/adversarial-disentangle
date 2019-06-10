import torch
from torch import nn
import torch.nn.functional as F

init_pstd = 0.541324855 # ln(e-1)
init_w = 1e-6
eps = 1e-8

def mul_no_back(x, k):
	return x + x.detach() * (k - 1)

class BatchRegularization(nn.Module):

	def __init__(self, num_features, alpha = 0.9):
		super(BatchRegularization, self).__init__()

		self.num_features = num_features
		self.alpha = alpha

		self.pre_mean = nn.Parameter(torch.zeros(num_features))
		self.pre_pstd = nn.Parameter(torch.Tensor(num_features).fill_(init_pstd))

		self.register_buffer('real_mean', torch.zeros(num_features))
		self.register_buffer('real_lstd', torch.zeros(num_features))
		self.register_buffer('real_w', torch.tensor(init_w))

		self.mode = 'disabled'
		self.loss = torch.tensor(0)

	def forward(self, input):
		if self.mode == 'real' or self.mode == 'fake':
			input_flat = input.contiguous().view(input.size(0), input.size(1), input.numel() // input.size(0) // input.size(1))
			pre_mean = self.pre_mean.unsqueeze(0).unsqueeze(2)
			pre_std = F.softplus(self.pre_pstd).unsqueeze(0).unsqueeze(2)
			input_transform = (input_flat - pre_mean).div(pre_std)

			batch_mean = input_transform.mean(2).mean(0)
			batch_ex2 = input_transform.pow(2).mean(2).mean(0)
			batch_lstd = (batch_ex2 - batch_mean.pow(2)).mul(input.size(0) / (input.size(0) - 1)).log()

			if self.mode == 'real':
				self.loss = (batch_mean.pow(2) + batch_lstd.pow(2)).mean()

				self.real_mean.data.mul(self.alpha).add_(batch_mean.data * (1 - self.alpha))
				self.real_lstd.data.mul(self.alpha).add_(batch_lstd.data * (1 - self.alpha))
				self.real_w.data.mul(self.alpha).add_(1 - self.alpha)
			else:
				real_mean = self.real_mean.div(self.real_w)
				real_lstd = self.real_lstd.div(self.real_w)

				self.loss = ((batch_mean - real_mean).pow(2) + (batch_lstd - real_lstd).pow(2)).mean()

		return input

	def get_loss(self):
		return self.loss

	def clear_loss(self):
		self.loss = torch.tensor(0)

	def set_mode(self, mode):
		self.clear_loss()
		self.mode = mode
