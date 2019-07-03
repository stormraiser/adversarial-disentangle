import random
import os
import os.path

import torch
from torch import optim, autograd
import torchvision.transforms as T

from . import Trainer
from datasets import ImageFolder, ParallelDataset
import models
from utils import *

copy_keys = ['mlp', 'batch_size', 'lr', 'sty_lr', 'lr_ramp',
	'rec_weight', 'cla_weight', 'con_weight', 'sty_weight', 'match_weight',
	'cla_br_weight', 'transform_penalty', 'augment_options',
	'nclass', 'vis_col']

class Stage1Trainer(Trainer):

	def __init__(self, options):
		super(Stage1Trainer, self).__init__(options, subfolders = ['reconstructions'], copy_keys = copy_keys)

		transforms = []
		if options.crop_size is not None:
			transforms.append(T.CenterCrop(options.crop_size))
		transforms.append(T.Resize(options.image_size))
		transforms.append(T.CenterCrop(options.image_size))
		transforms.append(T.ToTensor())

		image_transforms = transforms + [T.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1))]
		image_set = ImageFolder(options.data_root, transform = T.Compose(image_transforms))

		if options.weight_root is not None:
			self.has_weight = True
			weight_transforms = transforms + [lambda x: x[0]]
			weight_set = ImageFolder(options.weight_root, transform = T.Compose(weight_transforms))
			self.dataset = ParallelDataset(image_set, weight_set)
		else:
			self.has_weight = False
			self.dataset = image_set

		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = options.batch_size, shuffle = True, drop_last = True, num_workers = options.nloader)
		self.data_iter = iter(self.dataloader)

		self.enc = models.Encoder(options.image_size, options.image_size, options.enc_features, options.enc_blocks, options.enc_adain_features, options.enc_adain_blocks, options.content_size)
		self.enc.to(self.device)
		self.enc_optim = optim.Adam(self.enc.parameters(), lr = self.lr, eps = 1e-4)
		self.add_model('enc', self.enc, self.enc_optim)

		self.gen = models.TwoPartNestedDropoutGenerator(options.image_size, options.image_size, options.gen_features, options.gen_blocks, options.gen_adain_features, options.gen_adain_blocks, options.content_size, options.style_size)
		self.gen.to(self.device)
		self.gen_optim = optim.Adam(self.gen.parameters(), lr = self.lr, eps = 1e-4)
		self.add_model('gen', self.gen, self.gen_optim)

		if self.mlp:
			self.cla = models.MLPClassifier(options.content_size, options.mlp_features, options.mlp_layers, self.nclass)
		else:
			self.cla = models.ClassifierOrDiscriminator(options.image_size, options.image_size, options.cla_features, options.cla_blocks, options.cla_adain_features, options.cla_adain_blocks, self.nclass)
		self.cla.to(self.device)
		self.cla_optim = optim.Adam(self.cla.parameters(), lr = self.lr, eps = 1e-4)
		self.add_model('cla', self.cla, self.cla_optim)

		self.class_freq = image_set.get_class_freq().to(self.device)
		self.sty = models.NormalizedStyleBank(self.nclass, options.style_size, self.class_freq)
		self.sty.to(self.device)
		self.sty_optim = optim.Adam(self.sty.parameters(), lr = self.sty_lr, eps = 1e-8)
		self.add_model('sty', self.sty, self.sty_optim)

		if self.load_path is not None:
			self.vis_images = torch.load(os.path.join(self.load_path, 'reconstructions', 'images.pt'), map_location = self.device)
			self.vis_labels = torch.load(os.path.join(self.load_path, 'reconstructions', 'labels.pt'), map_location = self.device)
			if self.has_weight:
				self.vis_weights = torch.load(os.path.join(self.load_path, 'reconstructions', 'weights.pt'), map_location = self.device)
			self.load(options.load_iter)
		else:
			vis_images = []
			vis_labels = []
			if self.has_weight:
				vis_weights = []
			vis_index = random.sample(range(len(image_set)), options.vis_row * options.vis_col)
			for k in vis_index:
				image, label = image_set[k]
				vis_images.append(image)
				vis_labels.append(label)
				if self.has_weight:
					weight, _ = weight_set[k]
					vis_weights.append(weight)
			self.vis_images = torch.stack(vis_images, dim = 0).to(self.device)
			self.vis_labels = one_hot(torch.tensor(vis_labels, dtype = torch.int32), self.nclass).to(self.device)
			if self.has_weight:
				self.vis_weights = torch.stack(vis_weights, dim = 0).to(self.device)

		if self.save_path != self.load_path:
			torch.save(self.vis_images, os.path.join(self.save_path, 'reconstructions', 'images.pt'))
			torch.save(self.vis_labels, os.path.join(self.save_path, 'reconstructions', 'labels.pt'))
			save_image(self.vis_images.add(1).div(2), os.path.join(self.save_path, 'reconstructions', 'target.png'), self.vis_col)
			if self.has_weight:
				torch.save(self.vis_weights, os.path.join(self.save_path, 'reconstructions', 'weights.pt'))
				save_image(self.vis_weights.unsqueeze(1), os.path.join(self.save_path, 'reconstructions', 'weight.png'), self.vis_col)

		self.add_periodic_func(self.visualize_fixed, options.visualize_iter)
		self.visualize_fixed()

		self.con_drop_prob = torch.Tensor(options.content_size)
		for i in range(options.content_size):
			self.con_drop_prob[i] = options.content_dropout ** i
		self.sty_drop_prob = torch.Tensor(options.style_size)
		for i in range(options.style_size):
			self.sty_drop_prob[i] = options.style_dropout ** i

	def next_batch(self):
		try:
			batch = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.dataloader)
			batch = next(self.data_iter)

		if self.has_weight:
			images, labels, weights, _ = batch
		else:
			images, labels = batch
			weights = torch.ones_like(images[:, 0])

		return images.to(self.device), weights.to(self.device), one_hot(labels, self.nclass).to(self.device)

	def visualize(self, images, labels, filename, num_col):
		generated = []
		style_mean = []
		with torch.no_grad():
			for i in range((images.size(0) - 1) // self.batch_size + 1):
				batch_label = labels[i * self.batch_size : (i + 1) * self.batch_size]
				batch_style_mean, _ = self.sty(batch_label)
				style_mean.append(batch_style_mean)
			style_mean = torch.cat(style_mean, dim = 0)
			style_shift = torch.cat((style_mean[1:], style_mean[:1]), dim = 0)
			for i in range((images.size(0) - 1) // self.batch_size + 1):
				batch_sample = images[i * self.batch_size : (i + 1) * self.batch_size]
				batch_mean, _ = self.enc(alpha_mask(batch_sample))
				batch_style_mean = style_mean[i * self.batch_size : (i + 1) * self.batch_size]
				batch_style_shift = style_shift[i * self.batch_size : (i + 1) * self.batch_size]
				batch_rec = self.gen(batch_mean, batch_style_mean).cpu()
				batch_rec_zero = self.gen(batch_mean, torch.zeros_like(batch_style_mean)).cpu()
				batch_rec_shift = self.gen(batch_mean, batch_style_shift).cpu()
				generated.append(torch.stack((batch_rec, batch_rec_zero, batch_rec_shift), dim = 1))
			generated = torch.cat(generated, dim = 0).add(1).div(2)
		save_image(generated.view(images.size(0) * 3, 3, generated.size(3), generated.size(4)), filename, num_col * 3)

	def visualize_fixed(self):
		filename = os.path.join(self.save_path, 'reconstructions', '{0}.jpg'.format(self.state.iter))
		self.visualize(self.vis_images, self.vis_labels, filename, self.vis_col)

	def iter_func(self):
		lr_factor = min(self.state.iter / self.lr_ramp, 1)
		for group in self.enc_optim.param_groups:
			group['lr'] = self.lr * lr_factor
		for group in self.gen_optim.param_groups:
			group['lr'] = self.lr * lr_factor
		for group in self.cla_optim.param_groups:
			group['lr'] = self.lr * lr_factor
		for group in self.sty_optim.param_groups:
			group['lr'] = self.sty_lr * lr_factor

		images, weights, labels = self.next_batch()
		weights = images[:, 3].mul(weights)
		weights = weights.div(weights.sum(2, keepdim = True).sum(1, keepdim = True))

		self.enc.zero_grad()
		self.gen.zero_grad()
		self.cla.zero_grad()
		self.sty.zero_grad()

		con_mean, con_std = self.enc(alpha_mask(images))
		sty_mean, sty_std = self.sty(labels)

		con_drop_level = torch.rand(self.batch_size)
		con_drop_mask = torch.ge(self.con_drop_prob.unsqueeze(0), con_drop_level.unsqueeze(1)).float().to(self.device)
		sty_drop_level = torch.rand(self.batch_size)
		sty_drop_mask = torch.ge(self.sty_drop_prob.unsqueeze(0), sty_drop_level.unsqueeze(1)).float().to(self.device)

		con_code_loss = ((con_mean.pow(2) + con_std.pow(2)) * 0.5 - con_std.log() - 0.5).mean(0)
		con_code_loss_log = con_code_loss.sum().item()
		con_code_loss = con_code_loss.mul(self.con_drop_prob.to(self.device)).sum()
		con_code = sample_gaussian(con_mean, con_std)
		con_code_t = con_code.detach().requires_grad_()

		sty_code_loss = self.sty.get_code_loss()
		sty_code_loss_log = sty_code_loss.sum().item()
		sty_code_loss = sty_code_loss.mul(self.sty_drop_prob.to(self.device)).sum()
		sty_code = sample_gaussian(sty_mean, sty_std)
		sty_code_t = sty_code.detach().requires_grad_()

		rec = self.gen(con_code_t, sty_code_t, mask1 = con_drop_mask, mask2 = sty_drop_mask)
		rec_loss = (rec - images[:, :3]).pow(2).sum(1).add(1e-8).sqrt().mul(weights).sum(2).sum(1).mean()

		transform_penalty = self.gen.get_transform_penalty() or torch.tensor(0).to(self.device)
		self.gen.clear_transform_penalty()

		(rec_loss * self.rec_weight + transform_penalty * self.transform_penalty).backward()

		if self.cla_br_weight > 0:
			self.cla.set_batch_reg_mode('real')

		if self.mlp:
			con_code_t2 = con_code.detach().requires_grad_()
			cla_output = self.cla(con_code_t2, mask = con_drop_mask)
			rec_shift_loss = torch.tensor(0)
		else:
			sty_code_shift = torch.cat((sty_code_t[1:], sty_code_t[:1]), dim = 0).detach()
			sty_drop_mask_shift = torch.cat((sty_drop_mask[1:], sty_drop_mask[:1]), dim = 0)
			rec_shift = self.gen(con_code_t, sty_code_shift, mask1 = con_drop_mask, mask2 = sty_drop_mask_shift)
			rec_shift_t = rec_shift.detach().requires_grad_()
			cla_output = self.cla(augment(alpha_mask(rec_shift_t), self.augment_options, generate_aug_params(self.batch_size)))

			rec_shift_loss = (rec_shift_t - images[:, :3]).pow(2).sum(1).add(1e-8).sqrt().mul(weights).sum(2).sum(1).mean()
		
		cla_loss = -torch.mul(cla_output, labels).sum(1).mean(0)
		cla_adv_loss = -torch.mul(torch.max(cla_output, self.class_freq.log()), labels).sum(1).mean(0)

		cla_br_loss = self.cla.get_batch_reg_loss() if self.cla_br_weight > 0 else torch.tensor(0).to(self.device)
		self.cla.set_batch_reg_mode('disabled')

		(cla_loss + cla_br_loss * self.cla_br_weight).backward(retain_graph = True)

		if self.mlp:
			con_code_grad = con_code_t.grad + autograd.grad(-cla_adv_loss * self.cla_weight, con_code_t2)[0]
		else:
			rec_shift.backward(autograd.grad(-cla_adv_loss * self.cla_weight, rec_shift_t)[0])
			con_code_grad = con_code_t.grad

		autograd.backward([con_code, sty_code, con_code_loss * self.con_weight + sty_code_loss * self.sty_weight], [con_code_grad, sty_code_t.grad, None])

		if self.match_weight > 0:
			rec_shift = self.gen(con_code.detach(), sty_code_shift, mask1 = con_drop_mask, mask2 = sty_drop_mask_shift)
			rec_shift_t = rec_shift.detach().requires_grad_()
			with torch.no_grad():
				enc_output, _ = self.enc(alpha_mask(rec, images[:, 3]))
			enc_output_shift, _ = self.enc(alpha_mask(rec_shift_t, images[:, 3]))
			match_loss = (enc_output - enc_output_shift).pow(2).sum(1).mean()
			rec_shift.backward(autograd.grad(match_loss * self.match_weight, rec_shift_t)[0])
		else:
			match_loss = torch.tensor(0)

		self.enc_optim.step()
		self.gen_optim.step()
		self.cla_optim.step()
		self.sty_optim.step()

		self.log('con', con_code_loss_log)
		self.log('sty', sty_code_loss_log)
		self.log('rec', rec_loss.item())
		self.log('cla', cla_loss.item())
		self.log('cla-br', cla_br_loss.item())
		self.log('rec-shift', rec_shift_loss.item())
		self.log('match', match_loss.item())

		print('Iteration {0}:'.format(self.state.iter))
		print('rec:{0:.4f} con:{1:.2f} sty:{2:.2f} cla:{3:.2f} rec-shift:{4:.4f} match:{5:.2f}'.format(rec_loss.item(), con_code_loss_log, sty_code_loss_log, cla_loss.item(), rec_shift_loss.item(), match_loss.item()))
