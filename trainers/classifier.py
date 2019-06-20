import os
import os.path

import torch
from torch import optim, autograd
import torchvision.transforms as T

from . import Trainer
from datasets import ImageFolder
import models
from utils import *

copy_keys = ['nclass', 'lr', 'lr_ramp', 'batch_size', 'augment_options', 'cla_br_weight']

class ClassifierTrainer(Trainer):

	def __init__(self, options):
		super(ClassifierTrainer, self).__init__(options, copy_keys = copy_keys)

		transforms = []
		if options.crop_size is not None:
			transforms.append(T.CenterCrop(options.crop_size))
		transforms.append(T.Resize(options.image_size))
		transforms.append(T.CenterCrop(options.image_size))
		transforms.append(T.ToTensor())

		image_transforms = transforms + [T.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1))]
		self.dataset = ImageFolder(options.data_root, transform = T.Compose(image_transforms))
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = options.batch_size, shuffle = True, drop_last = True, num_workers = options.nloader)
		self.data_iter = iter(self.dataloader)

		self.cla = models.ClassifierOrDiscriminator(options.image_size, options.image_size, options.cla_features, options.cla_blocks, options.cla_adain_features, options.cla_adain_blocks, self.nclass)
		self.cla.to(self.device)
		self.cla_optim = optim.Adam(self.cla.parameters(), lr = self.lr, eps = 1e-4)
		self.add_model('cla', self.cla, self.cla_optim)

		if self.load_path is not None:
			self.load(options.load_iter)

	def next_batch(self):
		try:
			images, labels = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.dataloader)
			images, labels = next(self.data_iter)

		return images.to(self.device), one_hot(labels, self.nclass).to(self.device)

	def iter_func(self):
		lr_factor = min(self.state.iter / self.lr_ramp, 1)
		for group in self.cla_optim.param_groups:
			group['lr'] = self.lr * lr_factor

		images, labels = self.next_batch()

		self.cla.zero_grad()

		if self.cla_br_weight > 0:
			self.cla.set_batch_reg_mode('real')

		cla_output = self.cla(augment(alpha_mask(images), self.augment_options, generate_aug_params(self.batch_size)))

		cla_loss = -torch.mul(cla_output, labels).sum(1).mean(0)

		cla_br_loss = self.cla.get_batch_reg_loss() if self.cla_br_weight > 0 else torch.tensor(0).to(self.device)
		self.cla.set_batch_reg_mode('disabled')

		(cla_loss + cla_br_loss * self.cla_br_weight).backward()

		self.cla_optim.step()

		self.log('cla', cla_loss.item())
		self.log('cla-br', cla_br_loss.item())

		print('Iteration {0}: loss: {1:.4f} br: {2:.4f}'.format(self.state.iter, cla_loss.item(), cla_br_loss.item()))
