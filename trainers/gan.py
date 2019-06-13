import random
import os
import os.path

import torch
from torch import optim, autograd
import torchvision.transforms as T

from . import Trainer
from datasets import ImageFolder
import models
from utils import *

copy_keys = ['batch_size', 'lr', 'lr_ramp', 'augment_options', 'vis_col',
	'dis_br_weight', 'gen_br_weight', 'transform_penalty', 'content_size', 'force_dis']

class GANTrainer(Trainer):

	def __init__(self, options):
		super(GANTrainer, self).__init__(options, subfolders = ['samples'], copy_keys = copy_keys)

		transforms = []
		if options.crop_size is not None:
			transforms.append(T.CenterCrop(options.crop_size))
		transforms.append(T.Resize(options.image_size))
		transforms.append(T.ToTensor())
		transforms.append(T.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1)))

		self.dataset = ImageFolder(options.data_root, transform = T.Compose(transforms))
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = options.batch_size, shuffle = True, drop_last = True, num_workers = options.nloader)
		self.data_iter = iter(self.dataloader)

		self.gen = models.Generator(options.image_size, options.image_size, options.gen_features, options.gen_blocks, options.gen_adain_features, options.gen_adain_blocks, options.content_size)
		self.gen.to(self.device)
		self.gen_optim = optim.RMSprop(self.gen.parameters(), lr = self.lr, eps = 1e-4, alpha = 0.9)
		self.add_model('gen', self.gen, self.gen_optim)

		self.dis = models.ClassifierOrDiscriminator(options.image_size, options.image_size, options.dis_features, options.dis_blocks, options.dis_adain_features, options.dis_adain_blocks)
		self.dis.to(self.device)
		self.dis_optim = optim.RMSprop(self.dis.parameters(), lr = self.lr, eps = 1e-4, alpha = 0.9)
		self.add_model('dis', self.dis, self.dis_optim)

		if self.load_path is not None:
			self.vis_codes = torch.load(os.path.join(self.load_path, 'samples', 'codes.pt'), map_location = self.device)
			self.load(options.load_iter)
		else:
			self.vis_codes = gaussian_noise(options.vis_row * options.vis_col, options.content_size).to(self.device)
			self.state.dis_total_batches = 0

		if self.save_path != self.load_path:
			torch.save(self.vis_codes, os.path.join(self.save_path, 'samples', 'codes.pt'))

		self.add_periodic_func(self.visualize_fixed, options.visualize_iter)
		self.visualize_fixed()

		self.loss_avg_factor = 0.9

	def next_batch(self):
		try:
			images, _ = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.dataloader)
			images, _ = next(self.data_iter)

		return images.to(self.device)

	def visualize(self, codes, filename, num_col):
		generated = []
		with torch.no_grad():
			for i in range((codes.size(0) - 1) // self.batch_size + 1):
				batch_code = codes[i * self.batch_size : (i + 1) * self.batch_size]
				generated.append(self.gen(batch_code))
			generated = torch.cat(generated, dim = 0).add(1).div(2)
		save_image(generated, filename, num_col)

	def visualize_fixed(self):
		filename = os.path.join(self.save_path, 'samples', '{0}.jpg'.format(self.state.iter))
		self.visualize(self.vis_codes, filename, self.vis_col)

	def iter_func(self):
		dis_batches = 0
		dis_real_exp_avg = 0
		dis_fake_exp_avg = 0
		dis_br_exp_avg = 0

		while True:
			dis_batches += 1
			dis_lr_factor = min((self.state.dis_total_batches + dis_batches) / self.lr_ramp, 1)
			for group in self.dis_optim.param_groups:
				group['lr'] = self.lr * dis_lr_factor

			images = self.next_batch()

			self.dis.zero_grad()

			if self.dis_br_weight > 0 or self.gen_br_weight > 0:
				self.dis.set_batch_reg_mode('real')

			real_aug_params = generate_aug_params(self.batch_size)
			dis_real_loss = (self.dis(augment(alpha_mask(images), self.augment_options, real_aug_params)) - 1).pow(2).mean()
			dis_real_exp_avg = dis_real_exp_avg * self.loss_avg_factor + dis_real_loss.item() * (1 - self.loss_avg_factor)

			dis_br_loss = self.dis.get_batch_reg_loss() if self.dis_br_weight > 0 else torch.tensor(0).to(self.device)
			self.dis.set_batch_reg_mode('disabled')

			dis_br_exp_avg = dis_br_exp_avg * self.loss_avg_factor + dis_br_loss.item() * (1 - self.loss_avg_factor)

			(dis_real_loss + dis_br_loss * self.dis_br_weight).backward()

			rand_codes = gaussian_noise(self.batch_size, self.content_size).to(self.device)
			generated_t = self.gen(rand_codes).detach()

			fake_aug_params = generate_aug_params(self.batch_size)
			dis_fake_loss = (self.dis(augment(alpha_mask(generated_t, images[:, 3]), self.augment_options, fake_aug_params)) + 1).pow(2).mean()
			dis_fake_exp_avg = dis_fake_exp_avg * self.loss_avg_factor + dis_fake_loss.item() * (1 - self.loss_avg_factor)

			dis_fake_loss.backward()

			self.dis_optim.step()

			w = 1 - self.loss_avg_factor ** dis_batches
			if self.force_dis <= 0 or (dis_real_exp_avg / w < self.force_dis and dis_fake_exp_avg / w < self.force_dis):
				self.log('dis-real', dis_real_exp_avg / w)
				self.log('dis-fake', dis_fake_exp_avg / w)
				self.log('dis-br', dis_br_exp_avg / w)
				self.log('dis-batches', dis_batches)
				break

			if dis_batches % 10 == 0:
				print('{0} dis batches... d-r:{1:.4f} d-f:{2:.4f}'.format(dis_batches, dis_real_exp_avg / w, dis_fake_exp_avg / w))
		self.state.dis_total_batches += dis_batches
		
		gen_lr_factor = min(self.state.iter / self.lr_ramp, 1)
		for group in self.gen_optim.param_groups:
			group['lr'] = self.lr * gen_lr_factor

		self.gen.zero_grad()

		generated = self.gen(rand_codes)
		generated_t = generated.detach().requires_grad_()

		if self.gen_br_weight > 0:
			self.dis.set_batch_reg_mode('fake')

		gen_dis_output = self.dis(augment(alpha_mask(generated_t, images[:, 3]), self.augment_options, fake_aug_params))
		gen_loss = gen_dis_output.pow(2).mean()

		gen_br_loss = self.dis.get_batch_reg_loss()
		self.dis.set_batch_reg_mode('disabled')

		transform_penalty = self.gen.get_transform_penalty()
		self.gen.clear_transform_penalty()

		generated_grad = autograd.grad([gen_loss + gen_br_loss * self.gen_br_weight], [generated_t])[0]
		if (self.transform_penalty > 0) and (transform_penalty > 0):
			torch.autograd.backward([generated, transform_penalty * self.transform_penalty], [generated_grad, None])
		else:
			generated.backward(generated_grad)

		self.gen_optim.step()

		self.log('gen', gen_loss.item())
		self.log('gen-br', gen_br_loss.item())

		print('Iteration {0}: dis batches: {1} ({2})'.format(self.state.iter, dis_batches, self.state.dis_total_batches))
		print('d-r:{0:.4f} d-f:{1:.4f} g:{2:.4f} d-br:{3:.4f} g-br:{4:.4f}'.format(dis_real_exp_avg / w, dis_fake_exp_avg / w, gen_loss.item(), dis_br_exp_avg / w, gen_br_loss.item()))
