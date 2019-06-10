import torch
from torch import nn
import torch.nn.functional as F

from .ModelBase import ModelBase
import modules
from utils import compute_stats

class EncoderBase(ModelBase):

	def __init__(self, w_in, h_in, num_features, blocks, adain_features = -1, adain_blocks = -1, in_features = 4, batch_reg = False):
		super(EncoderBase, self).__init__(w_in, h_in, len(num_features) - 1)

		self.has_adain = (adain_features > 0)

		f_prev = in_features
		adain_total = 0
		self.levels = nn.ModuleList()
		for i in range(len(num_features) - 1):
			level = []
			f = num_features[i]
			for j, btype in enumerate(blocks[i]):
				if j == 0:
					if btype == 'c':
						level.append(modules.ResidueBlock(f_prev, f, 2, self.pad_h[i], self.pad_w[i], residue_ratio = self.next_residue_ratio(len(blocks[i])), batch_reg = batch_reg))
					else:
						raise ValueError('Invalid block type')
				else:
					if btype == 'b':
						level.append(modules.BusBlock(f, f, residue_ratio = self.next_residue_ratio(len(blocks[i])), batch_reg = batch_reg))
					elif btype == 'n':
						level.append(modules.NonlocalBlock(f, min(f // 4, 64), min(f // 4, 64), max(f // 256, 1), residue_ratio = self.next_residue_ratio(len(blocks[i])), batch_reg = batch_reg))
					elif btype == 'c':
						level.append(modules.ResidueBlock(f, f, 1, 0, 0, residue_ratio = self.next_residue_ratio(len(blocks[i])), batch_reg = batch_reg))
					else:
						raise ValueError('Invalid block type')
			f_prev = f
			self.levels.append(nn.Sequential(*level))
			if self.has_adain:
				adain_total += num_features[i]

		self.top_size = f_prev * self.top_h * self.top_w
		f_prev = self.top_size
		fc = []
		if num_features[-1] > 0:
			for i in range(len(blocks[-1])):
				if i == 0:
					fc.append(modules.Linear(f_prev, num_features[-1]))
					if batch_reg:
						fc.append(modules.BatchRegularization(num_features[-1]))
					fc.append(modules.CPReLU(num_features[-1]))
				else:
					fc.append(modules.LinearResidueBlock(num_features[-1], residue_ratio = 1 / (i + 1), batch_reg = batch_reg))
				f_prev = num_features[-1]
		self.fc = nn.Sequential(*fc)

		if self.has_adain:
			adain_map = []
			for i in range(adain_blocks):
				if i == 0:
					adain_map.append(modules.Linear(adain_total * 2, adain_features))
					if batch_reg:
						adain_map.append(modules.BatchRegularization(adain_features))
					adain_map.append(modules.CPReLU(adain_features))
				else:
					adain_map.append(modules.LinearResidueBlock(adain_features, residue_ratio = 1 / (i + 1), batch_reg = batch_reg))
			self.adain_map = nn.Sequential(*adain_map)
			self.out_features = f_prev + (adain_features if adain_blocks > 0 else (adain_total * 2))
		else:
			self.out_features = f_prev

	def forward(self, input):
		last = input
		if self.has_adain:
			level_stats = []
		for level in self.levels:
			last = level(last)
			if self.has_adain:
				mean, std = compute_stats(last)
				level_stats.append(mean)
				level_stats.append(std)
				last = last.sub(mean.unsqueeze(2).unsqueeze(3)).div(std.unsqueeze(2).unsqueeze(3))
		last = self.fc(last.contiguous().view(input.size(0), self.top_size))
		if self.has_adain:
			stats = torch.cat(level_stats, dim = 1)
			last = torch.cat((last, self.adain_map(stats)), dim = 1)
		return last

class Encoder(EncoderBase):

	def __init__(self, w_in, h_in, num_features, blocks, adain_features, adain_blocks, code_size):
		super(Encoder, self).__init__(w_in, h_in, num_features, blocks, adain_features, adain_blocks)
		self.mean = modules.Linear(self.out_features, code_size)
		self.pstd = modules.Linear(self.out_features, code_size)

	def forward(self, input):
		last = super(Encoder, self).forward(input)
		return self.mean(last), F.softplus(self.pstd(last))

class ClassifierOrDiscriminator(EncoderBase):

	def __init__(self, w_in, h_in, num_features, blocks, adain_features, adain_blocks, num_classes = -1, in_features = 4):
		super(ClassifierOrDiscriminator, self).__init__(w_in, h_in, num_features, blocks, adain_features, adain_blocks, batch_reg = True, in_features = in_features)
		self.num_classes = num_classes
		if num_classes > 0:
			self.cla = modules.Linear(self.out_features, num_classes)
		else:
			self.dis = modules.Linear(self.out_features, 1)

	def convert(self):
		self.dis = modules.Linear(self.out_features, 1)
		self.dis.to(self.cla.weight.device)
		del self.cla
		self.num_classes = -1

	def forward(self, input):
		last = super(ClassifierOrDiscriminator, self).forward(input)
		if self.num_classes > 0:
			return F.log_softmax(self.cla(last), dim = 1)
		else:
			return self.dis(last).squeeze(1)
