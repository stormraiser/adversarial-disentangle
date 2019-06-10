import torch
from torch import nn
import torch.nn.functional as F

from .ModelBase import ModelBase
import modules
from utils import compute_stats

class Generator(ModelBase):

	def __init__(self, w_out, h_out, num_features, blocks, adain_features, adain_blocks, code_size, out_features = 3):
		super(Generator, self).__init__(w_out, h_out, len(num_features) - 1)

		self.has_adain = (adain_features > 0)
		self.has_fc = (num_features[-1] >= 0)

		if self.has_adain:
			self.adain_maps = nn.ModuleList()
			f_prev = code_size
			adain_init = []
			for i in range(adain_blocks):
				if i == 0:
					adain_init.append(modules.Linear(f_prev, adain_features))
					adain_init.append(modules.CPReLU(adain_features))
				else:
					adain_init.append(modules.LinearResidueBlock(adain_features, residue_ratio = 1 / (i + 1)))
				f_prev = adain_features
			self.adain_init = nn.Sequential(*adain_init)
			for i in range(len(num_features) - 1):
				self.adain_maps.append(modules.Linear(f_prev, num_features[len(num_features) - 2 - i] * 2))

		if self.has_fc:
			f_prev = code_size
			fc = []
			for i in range(len(blocks[-1])):
				if i == 0:
					fc.append(modules.Linear(f_prev, num_features[-1]))
					fc.append(modules.CPReLU(num_features[-1]))
				else:
					fc.append(modules.LinearResidueBlock(num_features[-1], residue_ratio = 1 / (i + 1)))
				f_prev = num_features[-1]
			fc.append(modules.Linear(f_prev, num_features[-2] * self.top_h * self.top_w))
			fc.append(modules.CPReLU(num_features[-2] * self.top_h * self.top_w))
			self.fc = nn.Sequential(*fc)
			self.top_size = (num_features[-2], self.top_h, self.top_w)
		else:
			self.top_value = nn.Parameter(torch.randn(num_features[-2], self.top_h, self.top_w))

		self.levels = nn.ModuleList()
		for i in range(len(num_features) - 1):
			depth = len(num_features) - 2 - i
			level = []
			f = num_features[depth]
			f_next = out_features if depth == 0 else num_features[depth - 1]
			for j, btype in enumerate(reversed(blocks[depth])):
				if j == len(blocks[depth]) - 1:
					if btype == 't':
						level.append(modules.ResidueBlock(f, f_next + 2, -2, self.pad_h[depth], self.pad_w[depth], residue_ratio = self.next_residue_ratio(len(blocks[depth])), nonlinear = (depth > 0)))
						level.append(modules.SpatialTransform())
					elif btype == 'c':
						level.append(modules.ResidueBlock(f, f_next, -2, self.pad_h[depth], self.pad_w[depth], residue_ratio = self.next_residue_ratio(len(blocks[depth])), nonlinear = (depth > 0)))
					else:
						raise ValueError('unknown block type')
				else:
					if btype == 'b':
						level.append(modules.BusBlock(f, f, residue_ratio = self.next_residue_ratio(len(blocks[depth]))))
					elif btype == 'n':
						level.append(modules.NonlocalBlock(f, min(f // 4, 64), min(f // 4, 64), max(f // 256, 1), residue_ratio = self.next_residue_ratio(len(blocks[depth]))))
					elif btype == 't':
						level.append(modules.ResidueBlock(f, f + 2, 1, 0, 0, residue_ratio = self.next_residue_ratio(len(blocks[depth]))))
						level.append(modules.SpatialTransform())
					elif btype == 'c':
						level.append(modules.ResidueBlock(f, f, 1, 0, 0, residue_ratio = self.next_residue_ratio(len(blocks[depth]))))
					else:
						raise ValueError('unknown block type')
			self.levels.append(nn.Sequential(*level))

	def forward(self, input):
		if self.has_adain:
			adain_out = self.adain_init(input)
		if self.has_fc:
			last = self.fc(input).contiguous().view(input.size(0), *self.top_size)
		else:
			last = self.top_value.unsqueeze(0).expand_as(input.size(0), *self.top_value.size())
		for i in range(len(self.levels)):
			if self.has_adain:
				adain_param = self.adain_maps[i](adain_out)
				scale = F.softplus(adain_param[:, :adain_param.size(1) // 2]).unsqueeze(2).unsqueeze(3)
				bias = adain_param[:, adain_param.size(1) // 2:].unsqueeze(2).unsqueeze(3)
				mean, std = compute_stats(last)
				last = last.sub(mean.unsqueeze(2).unsqueeze(3)).div(std.unsqueeze(2).unsqueeze(3)).mul(scale).add(bias)
			last = self.levels[i](last)
		return last.tanh()

class NestedDropoutGenerator(Generator):

	def __init__(self, w_out, h_out, num_features, blocks, adain_features, adain_blocks, code_size):
		super(NestedDropoutGenerator, self).__init__(w_out, h_out, num_features, blocks, adain_features, adain_blocks, code_size)

	def forward(self, input, mask):
		if mask is None:
			mask = torch.ones_like(input)
		input = input.mul(mask)
		return super(NestedDropoutGenerator, self).forward(input)

class TwoPartNestedDropoutGenerator(Generator):

	def __init__(self, w_out, h_out, num_features, blocks, adain_features, adain_blocks, code_size1, code_size2):
		super(TwoPartNestedDropoutGenerator, self).__init__(w_out, h_out, num_features, blocks, adain_features, adain_blocks, code_size1 + code_size2)

	def forward(self, input1, input2, mask1 = None, mask2 = None):
		if mask1 is None:
			mask1 = torch.ones_like(input1)
		input1 = input1.mul(mask1)
		if mask2 is None:
			mask2 = torch.ones_like(input2)
		input2 = input2.mul(mask2)
		return super(TwoPartNestedDropoutGenerator, self).forward(torch.cat((input1, input2), dim = 1))
