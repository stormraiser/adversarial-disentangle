import torch
from torch import nn
import torch.nn.functional as F

class SpatialTransform(nn.Module):

	def __init__(self):
		super(SpatialTransform, self).__init__()

		self.penalty = torch.tensor(0)

	def forward(self, input):
		value = input[:, :-2]
		offset = input[:, -2:]

		offset_xmax = F.max_pool2d(offset[:, :1], (2, 1), 1)
		offset_xmin = -F.max_pool2d(-offset[:, :1], (2, 1), 1)
		offset_ymax = F.max_pool2d(offset[:, 1:], (1, 2), 1)
		offset_ymin = -F.max_pool2d(-offset[:, 1:], (1, 2), 1)
		offset_xdiff = offset_xmin[:, :, :, 1:] + 1 - offset_xmax[:, :, :, :-1]
		offset_ydiff = offset_ymin[:, :, 1:] + 1 - offset_ymax[:, :, :-1]
		self.penalty = -(offset_xdiff.clamp(max = 0) + offset_ydiff.clamp(max = 0)).mean()

		grid_basex = torch.linspace(-1, 1, value.size(3)).unsqueeze(0).unsqueeze(1).expand(value.size(0), value.size(2), value.size(3)).to(value)
		grid_basey = torch.linspace(-1, 1, value.size(2)).unsqueeze(0).unsqueeze(2).expand(value.size(0), value.size(2), value.size(3)).to(value)
		# 1 unit == 1/2 pixel length
		grid_x = (offset[:, 0] / value.size(3) + grid_basex).clamp(min = -1, max = 1)
		grid_y = (offset[:, 1] / value.size(2) + grid_basey).clamp(min = -1, max = 1)
		grid_coord = torch.stack((grid_x, grid_y), dim = 3)
		return F.grid_sample(value, grid_coord)

	def get_penalty(self):
		return self.penalty

	def clear_penalty(self):
		self.penalty = torch.tensor(0)
