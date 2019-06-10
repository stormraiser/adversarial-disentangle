import os
import os.path

import torch
import torchvision.transforms as T
from datasets import ImageFolder
import models

class FullDatasetEncoder:

	def __init__(self, options):
		self.device = options.device
		self.enc_path = options.enc_path
		self.content_size = options.content_size
		self.enc_iter = options.enc_iter

		if not os.path.exists(os.path.join(self.enc_path, 'codes')):
			os.makedirs(os.path.join(self.enc_path, 'codes'))

		transforms = []
		if options.crop_size is not None:
			transforms.append(T.CenterCrop(options.crop_size))
		transforms.append(T.Resize(options.image_size))
		transforms.append(T.ToTensor())
		transforms.append(T.Normalize((0.5, 0.5, 0.5, 0), (0.5, 0.5, 0.5, 1)))

		self.dataset = ImageFolder(options.data_root, transform = T.Compose(transforms))
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = options.batch_size, num_workers = options.nloader)
		self.data_iter = iter(self.dataloader)

		self.enc = models.Encoder(options.image_size, options.image_size, options.enc_features, options.enc_blocks, options.enc_adain_features, options.enc_adain_blocks, options.content_size)
		self.enc.to(self.device)
		self.enc.load_state_dict(torch.load(os.path.join(self.enc_path, 'models', '{0}_enc.pt'.format(self.enc_iter)), map_location = self.device))

	def run(self):
		codes = []

		nbatch = len(self.dataloader)
		n = len(self.dataset)
		m = self.content_size

		ex = torch.zeros(m, dtype = torch.double).to(self.device)
		exy = torch.zeros(m, m, dtype = torch.double).to(self.device)
		es2 = torch.zeros(m, dtype = torch.double).to(self.device)

		with torch.no_grad():
			for k, batch in enumerate(self.data_iter):
				print('batch {0} / {1}'.format(k + 1, nbatch))

				batch_mean, batch_std = self.enc(batch[0].to(self.device))
				codes.append(torch.stack((batch_mean, batch_std), dim = 1).cpu())

				batch_mean = batch_mean.double()
				batch_std = batch_std.double()

				ex.add_(batch_mean.sum(0))
				exy.add_(torch.matmul(batch_mean.transpose(0, 1), batch_mean))
				es2.add_(batch_std.pow(2).sum(0))

		ex = ex.cpu().float() / n
		exy = exy.cpu().float() / n
		es2 = es2.cpu().float() / n

		mean_var = torch.diag(exy) - ex.pow(2)
		full_var = mean_var + es2
		var_ratio = mean_var.div(full_var)

		full_cxy = (exy + torch.diag(es2) - torch.mul(ex.unsqueeze(0), ex.unsqueeze(1)))
		eigval, eigvec = torch.symeig(full_cxy, eigenvectors = True)

		stats = {
			'full_mean' : ex,
			'full_std' : full_var.sqrt(),
			'eigval' : eigval,
			'eigvec' : eigvec,
			'dim_weight' : var_ratio
		}

		codes = torch.cat(codes, dim = 0)
		torch.save(codes, os.path.join(self.enc_path, 'codes', '{0}_codes.pt'.format(self.enc_iter)))
		torch.save(stats, os.path.join(self.enc_path, 'codes', '{0}_stats.pt'.format(self.enc_iter)))
