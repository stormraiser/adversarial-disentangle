import torch
from torch import nn

import modules

class ModelBase(nn.Module):

	def next_residue_ratio(self, level_length):
		layer_weight = 1 / level_length
		self.total_weight += layer_weight
		return layer_weight / self.total_weight

	def set_batch_reg_mode(self, mode):
		for module in self.modules():
			if isinstance(module, modules.BatchRegularization):
				module.set_mode(mode)

	def get_batch_reg_loss(self):
		loss = 0
		for module in self.modules():
			if isinstance(module, modules.BatchRegularization):
				loss = loss + module.get_loss()
		return loss

	def clear_batch_reg_loss(self):
		for module in self.modules():
			if isinstance(module, modules.BatchRegularization):
				module.clear_loss()

	def get_transform_penalty(self):
		penalty = 0
		for module in self.modules():
			if isinstance(module, modules.SpatialTransform):
				penalty = penalty + module.get_penalty()
		return penalty

	def clear_transform_penalty(self):
		for module in self.modules():
			if isinstance(module, modules.SpatialTransform):
				module.clear_penalty()

	def __init__(self, width, height, num_conv_levels):
		super(ModelBase, self).__init__()

		self.total_weight = 1

		self.pad_w = []
		self.pad_h = []
		for i in range(num_conv_levels - 1):
			self.pad_w.append((width % 4) // 2)
			width = (width + 2) // 4 * 2
			self.pad_h.append((height % 4) // 2)
			height = (height + 2) // 4 * 2
		self.pad_w.append(0)
		self.pad_h.append(0)
		self.top_w = width // 2
		self.top_h = height // 2
