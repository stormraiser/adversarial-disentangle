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

copy_keys = ['batch_size', 'lr', 'sty_lr', 'lr_ramp',
	'dis_weight', 'cla_weight', 'con_weight', 'sty_weight',
	'cla_br_weight', 'dis_br_weight', 'gen_br_weight',
	'transform_penalty', 'augment_options',
	'nclass', 'vis_col', 'cla_fake', 'force_dis', 'force_cla']

class Stage2Trainer(Trainer):

	def __init__(self, options):
		super(Stage2Trainer, self).__init__(options, subfolders = ['samples', 'reconstructions'], copy_keys = copy_keys)

		transforms = []
		if options.crop_size is not None:
			transforms.append(T.CenterCrop(options.crop_size))
		transforms.append(T.Resize(options.image_size))
		transforms.append(T.ToTensor())
		transforms.append(T.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1)))

		image_set = ImageFolder(options.data_root, transform = T.Compose(transforms))
		enc_codes = torch.load(os.path.join(options.enc_path, 'codes', '{0}_codes.pt'.format(options.enc_iter)))
		code_set = torch.utils.data.TensorDataset(enc_codes[:, 0], enc_codes[:, 1])
		self.dataset = ParallelDataset(image_set, code_set)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = options.batch_size, shuffle = True, drop_last = True, num_workers = options.nloader)
		self.data_iter = iter(self.dataloader)

		enc_stats = torch.load(os.path.join(options.enc_path, 'codes', '{0}_stats.pt'.format(options.enc_iter)))
		self.con_full_mean = enc_stats['full_mean']
		self.con_full_std = enc_stats['full_std']
		self.con_eigval = enc_stats['eigval']
		self.con_eigvec = enc_stats['eigvec']
		self.dim_weight = enc_stats['dim_weight']

		if self.con_weight > 0:
			self.enc = models.Encoder(options.image_size, options.image_size, options.enc_features, options.enc_blocks, options.enc_adain_features, options.enc_adain_blocks, options.content_size)
			self.enc.to(self.device)
			self.enc.load_state_dict(torch.load(os.path.join(options.enc_path, 'models', '{0}_enc.pt'.format(options.enc_iter)), map_location = self.device))

		self.gen = models.TwoPartNestedDropoutGenerator(options.image_size, options.image_size, options.gen_features, options.gen_blocks, options.gen_adain_features, options.gen_adain_blocks, options.content_size, options.style_size)
		self.gen.to(self.device)
		if (self.load_path is None) and not options.reset_gen:
			self.gen.load_state_dict(torch.load(os.path.join(options.enc_path, 'models', '{0}_gen.pt'.format(options.enc_iter)), map_location = self.device))
		self.gen_optim = optim.RMSprop(self.gen.parameters(), lr = self.lr, eps = 1e-4)
		self.add_model('gen', self.gen, self.gen_optim)

		self.cla = models.ClassifierOrDiscriminator(options.image_size, options.image_size, options.cla_features, options.cla_blocks, options.cla_adain_features, options.cla_adain_blocks, self.nclass)
		self.cla.to(self.device)
		if (self.load_path is None) and (options.cla_path is not None) and not options.reset_cla:
			self.cla.load_state_dict(torch.load(os.path.join(options.cla_path, 'models', '{0}_cla.pt'.format(options.cla_iter)), map_location = self.device))
		self.cla_optim = optim.RMSprop(self.cla.parameters(), lr = self.lr, eps = 1e-4)
		self.add_model('cla', self.cla, self.cla_optim)

		self.dis = models.ClassifierOrDiscriminator(options.image_size, options.image_size, options.dis_features, options.dis_blocks, options.dis_adain_features, options.dis_adain_blocks, self.nclass)
		self.dis.to(self.device)
		if (self.load_path is None) and (options.cla_path is not None) and not options.reset_dis:
			self.dis.load_state_dict(torch.load(os.path.join(options.cla_path, 'models', '{0}_cla.pt'.format(options.cla_iter)), map_location = self.device))
		self.dis.convert()
		self.dis_optim = optim.RMSprop(self.dis.parameters(), lr = self.lr, eps = 1e-4)
		self.add_model('dis', self.dis, self.dis_optim)

		self.sty = models.NormalizedStyleBank(self.nclass, options.style_size, image_set.get_class_freq())
		self.sty.to(self.device)
		if (self.load_path is None) and not options.reset_sty:
			self.sty.load_state_dict(torch.load(os.path.join(options.enc_path, 'models', '{0}_sty.pt'.format(options.enc_iter)), map_location = self.device))
		self.sty_optim = optim.Adam(self.sty.parameters(), lr = self.sty_lr, eps = 1e-8)
		self.add_model('sty', self.sty, self.sty_optim)

		if self.load_path is not None:
			rec_images = torch.load(os.path.join(self.load_path, 'reconstructions', 'images.pt'), map_location = self.device)
			self.rec_codes = torch.load(os.path.join(self.load_path, 'reconstructions', 'codes.pt'), map_location = self.device)
			self.rec_labels = torch.load(os.path.join(self.load_path, 'reconstructions', 'labels.pt'), map_location = self.device)
			self.vis_codes = torch.load(os.path.join(self.load_path, 'samples', 'codes.pt'), map_location = self.device)
			self.vis_style_noise = torch.load(os.path.join(self.load_path, 'samples', 'style_noise.pt'), map_location = self.device)
			self.load(options.load_iter)
		else:
			rec_images = []
			rec_codes = []
			rec_labels = []
			rec_index = random.sample(range(len(self.dataset)), options.vis_row * options.vis_col)
			for k in rec_index:
				image, label, code, _ = self.dataset[k]
				rec_images.append(image)
				rec_codes.append(code)
				rec_labels.append(label)
			rec_images = torch.stack(rec_images, dim = 0)
			self.rec_codes = torch.stack(rec_codes, dim = 0).to(self.device)
			self.rec_labels = one_hot(torch.tensor(rec_labels, dtype = torch.int32), self.nclass).to(self.device)
			self.vis_codes = self.noise_to_con_code(gaussian_noise(options.vis_row * options.vis_col, options.content_size)).to(self.device)
			self.vis_style_noise = gaussian_noise(options.vis_row * options.vis_col, options.style_size).to(self.device)

			self.state.dis_total_batches = 0

		if self.save_path != self.load_path:
			torch.save(rec_images, os.path.join(self.save_path, 'reconstructions', 'images.pt'))
			torch.save(self.rec_codes, os.path.join(self.save_path, 'reconstructions', 'codes.pt'))
			torch.save(self.rec_labels, os.path.join(self.save_path, 'reconstructions', 'labels.pt'))
			torch.save(self.vis_codes, os.path.join(self.save_path, 'samples', 'codes.pt'))
			torch.save(self.vis_style_noise, os.path.join(self.save_path, 'samples', 'style_noise.pt'))
			save_image(rec_images.add(1).div(2), os.path.join(self.save_path, 'reconstructions', 'target.png'), self.vis_col)

		self.add_periodic_func(self.visualize_fixed, options.visualize_iter)
		self.visualize_fixed()

		self.loss_avg_factor = 0.9

		self.sty_drop_prob = torch.Tensor(options.style_size)
		for i in range(options.style_size):
			self.sty_drop_prob[i] = options.style_dropout ** i

	def next_batch(self):
		try:
			images, labels, mean, std = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.dataloader)
			images, labels, mean, std = next(self.data_iter)

		return images.to(self.device), one_hot(labels, self.nclass).to(self.device), mean.to(self.device), std.to(self.device)

	def normalize_con_code(self, code):
		return code.sub(self.con_full_mean.unsqueeze(0).to(code)).div(self.con_full_std.unsqueeze(0).to(code))

	def noise_to_con_code(self, noise):
		coef = torch.matmul(noise, self.con_eigvec.to(noise))
		return torch.matmul(coef.mul(self.con_eigval.sqrt().unsqueeze(0).to(noise)), self.con_eigvec.transpose(0, 1).to(noise)).add(self.con_full_mean.unsqueeze(0).to(noise))

	def noise_to_sty_code(self, noise):
		sty_cov = self.sty.get_normalized_cov().cpu()
		sty_eigval, sty_eigvec = torch.symeig(sty_cov, eigenvectors = True)
		coef = torch.matmul(noise, sty_eigvec.to(noise))
		return torch.matmul(coef.mul(sty_eigval.sqrt().unsqueeze(0).to(noise)), sty_eigvec.transpose(0, 1).to(noise))

	def visualize(self, con_codes, sty_codes, filename, num_col):
		generated = []
		sty_code_shift = torch.cat((sty_codes[1:], sty_codes[:1]), dim = 0)
		with torch.no_grad():
			for i in range((con_codes.size(0) - 1) // self.batch_size + 1):
				batch_con = con_codes[i * self.batch_size : (i + 1) * self.batch_size]
				batch_sty = sty_codes[i * self.batch_size : (i + 1) * self.batch_size]
				batch_sty_shift = sty_code_shift[i * self.batch_size : (i + 1) * self.batch_size]
				batch_gen = self.gen(batch_con, batch_sty).cpu()
				batch_gen_zero = self.gen(batch_con, torch.zeros_like(batch_sty)).cpu()
				batch_gen_shift = self.gen(batch_con, batch_sty_shift).cpu()
				generated.append(torch.stack((batch_gen, batch_gen_zero, batch_gen_shift), dim = 1))
			generated = torch.cat(generated, dim = 0).add(1).div(2)
		save_image(generated.view(con_codes.size(0) * 3, 3, generated.size(3), generated.size(4)), filename, num_col * 3)

	def visualize_fixed(self):
		vis_filename = os.path.join(self.save_path, 'samples', '{0}.jpg'.format(self.state.iter))
		vis_styles = self.noise_to_sty_code(self.vis_style_noise)
		self.visualize(self.vis_codes, vis_styles, vis_filename, self.vis_col)
		rec_filename = os.path.join(self.save_path, 'reconstructions', '{0}.jpg'.format(self.state.iter))
		rec_styles, _ = self.sty(self.rec_labels)
		self.visualize(self.rec_codes, rec_styles, rec_filename, self.vis_col)

	def iter_func(self):
		dis_batches = 0
		dis_real_exp_avg = 0
		dis_fake_exp_avg = 0
		dis_br_exp_avg = 0
		cla_real_exp_avg = 0
		cla_fake_exp_avg = 0
		cla_br_exp_avg = 0

		while True:
			dis_batches += 1
			dis_lr_factor = min((self.state.dis_total_batches + dis_batches) / self.lr_ramp, 1)
			for group in self.dis_optim.param_groups:
				group['lr'] = self.lr * dis_lr_factor
			for group in self.cla_optim.param_groups:
				group['lr'] = self.lr * dis_lr_factor

			images, labels, con_mean, con_std = self.next_batch()
			label_shift = torch.cat((labels[1:], labels[:1]), dim = 0)
			con_code = sample_gaussian(con_mean, con_std)

			aug_params = generate_aug_params(self.batch_size)
			real_aug = augment(images, self.augment_options, aug_params)

			sty_drop_level = torch.rand(self.batch_size)
			sty_drop_mask = torch.ge(self.sty_drop_prob.unsqueeze(0), sty_drop_level.unsqueeze(1)).float().to(self.device)
			sty_drop_mask_shift = torch.cat((sty_drop_mask[1:], sty_drop_mask[:1]), dim = 0)

			sty_mean, sty_std = self.sty(labels)
			sty_code = sample_gaussian(sty_mean, sty_std)
			sty_code_shift = torch.cat((sty_code[1:], sty_code[:1]), dim = 0)

			generated_t = self.gen(con_code, sty_code_shift, mask2 = sty_drop_mask_shift).detach()
			generated_aug = augment(alpha_mask(generated_t, images[:, 3]), self.augment_options, aug_params)

			self.dis.zero_grad()

			if self.dis_br_weight > 0 or self.gen_br_weight > 0:
				self.dis.set_batch_reg_mode('real')

			dis_real_loss = (self.dis(real_aug) - 1).pow(2).mean()
			dis_real_exp_avg = dis_real_exp_avg * self.loss_avg_factor + dis_real_loss.item() * (1 - self.loss_avg_factor)

			dis_br_loss = self.dis.get_batch_reg_loss() if self.dis_br_weight > 0 else torch.tensor(0).to(self.device)
			self.dis.set_batch_reg_mode('disabled')
			dis_br_exp_avg = dis_br_exp_avg * self.loss_avg_factor + dis_br_loss.item() * (1 - self.loss_avg_factor)

			(dis_real_loss + dis_br_loss * self.dis_br_weight).backward()

			dis_fake_loss = (self.dis(generated_aug) + 1).pow(2).mean()
			dis_fake_exp_avg = dis_fake_exp_avg * self.loss_avg_factor + dis_fake_loss.item() * (1 - self.loss_avg_factor)
			dis_fake_loss.backward()

			self.dis_optim.step()

			self.cla.zero_grad()

			if self.cla_br_weight > 0 or self.gen_br_weight > 0:
				self.cla.set_batch_reg_mode('real')

			cla_real_loss = -torch.mul(self.cla(real_aug), labels).sum(1).mean()
			cla_real_exp_avg = cla_real_exp_avg * self.loss_avg_factor + cla_real_loss.item() * (1 - self.loss_avg_factor)

			cla_br_loss = self.cla.get_batch_reg_loss() if self.cla_br_weight > 0 else torch.tensor(0).to(self.device)
			self.cla.set_batch_reg_mode('disabled')
			cla_br_exp_avg = cla_br_exp_avg * self.loss_avg_factor + cla_br_loss.item() * (1 - self.loss_avg_factor)

			(cla_real_loss + cla_br_loss * self.cla_br_weight).backward()

			if self.cla_fake:
				cla_fake_loss = -torch.mul((1 - self.cla(generated_aug).exp()).add(1e-6).log(), label_shift).sum(1).mean()
				cla_fake_exp_avg = cla_fake_exp_avg * self.loss_avg_factor + cla_fake_loss.item() * (1 - self.loss_avg_factor)
				cla_fake_loss.backward()

			self.cla_optim.step()

			w = 1 - self.loss_avg_factor ** dis_batches
			dis_flag = self.force_dis <= 0 or (dis_real_exp_avg / w < self.force_dis and dis_fake_exp_avg / w < self.force_dis)
			cla_flag = self.force_cla <= 0 or (cla_real_exp_avg / w < self.force_cla and cla_fake_exp_avg / w < self.force_cla)
			if dis_flag and cla_flag:
				self.log('dis-real', dis_real_exp_avg / w)
				self.log('dis-fake', dis_fake_exp_avg / w)
				self.log('dis-br', dis_br_exp_avg / w)
				self.log('dis-batches', dis_batches)
				self.log('cla-real', cla_real_exp_avg / w)
				self.log('cla-fake', cla_fake_exp_avg / w)
				self.log('cla-br', cla_br_exp_avg / w)
				break

			if dis_batches % 10 == 0:
				print('{0} dis batches... dr:{1:.4f} df:{2:.4f} cr:{3:.4f} cf:{4:.4f}'.format(dis_batches, dis_real_exp_avg / w, dis_fake_exp_avg / w, cla_real_exp_avg / w, cla_fake_exp_avg / w))
		self.state.dis_total_batches += dis_batches

		gen_lr_factor = min(self.state.iter / self.lr_ramp, 1)
		for group in self.gen_optim.param_groups:
			group['lr'] = self.lr * gen_lr_factor
		for group in self.sty_optim.param_groups:
			group['lr'] = self.sty_lr * gen_lr_factor

		self.gen.zero_grad()
		self.sty.zero_grad()

		sty_code_loss = self.sty.get_code_loss()
		sty_code_loss_log = sty_code_loss.sum().item()
		sty_code_loss = sty_code_loss.mul(self.sty_drop_prob.to(self.device)).sum()

		generated = self.gen(con_code, sty_code_shift, mask2 = sty_drop_mask_shift)
		generated_t = generated.detach().requires_grad_()
		generated_aug = augment(alpha_mask(generated_t, images[:, 3]), self.augment_options, aug_params)

		if self.gen_br_weight > 0:
			self.dis.set_batch_reg_mode('fake')
			self.cla.set_batch_reg_mode('fake')

		gen_dis_loss = self.dis(generated_aug).pow(2).mean()
		gen_cla_loss = -torch.mul(self.cla(generated_aug), label_shift).sum(1).mean()

		gen_dis_br_loss = self.dis.get_batch_reg_loss() if self.gen_br_weight > 0 else torch.tensor(0).to(self.device)
		gen_cla_br_loss = self.cla.get_batch_reg_loss() if self.gen_br_weight > 0 else torch.tensor(0).to(self.device)
		self.dis.set_batch_reg_mode('disabled')
		self.cla.set_batch_reg_mode('disabled')

		transform_penalty = self.gen.get_transform_penalty()
		self.gen.clear_transform_penalty()

		if self.con_weight > 0:
			enc_output, _ = self.enc(alpha_mask(generated_t, images[:, 3]))
			con_code_loss = (self.normalize_con_code(enc_output) - self.normalize_con_code(con_mean)).pow(2).mul(self.dim_weight.unsqueeze(0).to(self.device)).sum(1).mean()
		else:
			con_code_loss = torch.tensor(0).to(self.device)

		generated_grad = autograd.grad([(gen_dis_loss + gen_dis_br_loss * self.gen_br_weight) * self.dis_weight + (gen_cla_loss + gen_cla_br_loss * self.gen_br_weight) * self.cla_weight + con_code_loss * self.con_weight], [generated_t])[0]
		torch.autograd.backward([generated, sty_code_loss * self.sty_weight + transform_penalty * self.transform_penalty], [generated_grad, None])

		self.gen_optim.step()
		self.sty_optim.step()

		self.log('gen-dis', gen_dis_loss.item())
		self.log('gen-cla', gen_cla_loss.item())
		self.log('gen-dis-br', gen_dis_br_loss.item())
		self.log('gen-cla-br', gen_cla_br_loss.item())
		self.log('con', con_code_loss.item())
		self.log('sty', sty_code_loss_log)

		print('Iteration {0}: dis batches: {1} ({2})'.format(self.state.iter, dis_batches, self.state.dis_total_batches))
		print('dr:{0:.4f} df:{1:.4f} gd:{2:.4f} cr:{3:.4f} cf:{4:.4f} gc:{5:.4f} c:{6:.2f} s:{7:.2f}'.format(
			dis_real_exp_avg / w, dis_fake_exp_avg / w, gen_dis_loss.item(), cla_real_exp_avg / w, cla_fake_exp_avg / w, gen_cla_loss.item(), con_code_loss.item(), sty_code_loss_log))
