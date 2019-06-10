import math

import torch
import torch.nn.functional as F
import torchvision

def compute_stats(input):
	input_flat = input.view(input.size(0), input.size(1), input.numel() // input.size(0) // input.size(1))
	numel = input_flat.size(2)
	mean = input_flat.mean(2)
	std = (input_flat.pow(2).mean(2) - mean.pow(2) + 1e-8).sqrt()
	return mean, std

def alpha_mask(input, alpha = None):
	if alpha is None:
		alpha = input[:, 3] if input.size(1) == 4 else torch.ones_like(input[:, 0])
	return torch.cat((input[:, :3].mul(alpha.gt(0).float().unsqueeze(1)), alpha.unsqueeze(1)), dim = 1)

def one_hot(label, nclass):
	ret = torch.zeros(label.size(0), nclass)
	for i in range(label.size(0)):
		ret[i, label[i].item()] = 1
	return ret

def save_image(sample, filename, num_col, margin = 1):
	num_row = (sample.size(0) - 1) // num_col + 1
	vis_image = torch.zeros(sample.size(1), num_row * (sample.size(2) + margin * 2), num_col * (sample.size(3) + margin * 2))
	for i in range(num_row):
		for j in range(num_col):
			if i * num_col + j < sample.size(0):
				copy_sample = sample[i * num_col + j]
				vis_image[:, i * (sample.size(2) + margin * 2) + margin : (i + 1) * (sample.size(2) + margin * 2) - margin, j * (sample.size(3) + margin * 2) + margin : (j + 1) * (sample.size(3) + margin * 2) - margin].copy_(copy_sample)
	torchvision.utils.save_image(vis_image.clamp(min = 0, max = 1), filename, num_col)

def gaussian_noise(m, k):
	base = torch.stack([torch.randperm(m, dtype = torch.float32) for i in range(k)], dim = 1)
	rand = (base + torch.rand(m, k)) / m * 1.99998 - 0.99999
	return torch.erfinv(rand) * (2 ** 0.5)

def sample_gaussian(mean, std):
	return mean + std.mul(gaussian_noise(std.size(0), std.size(1)).to(std))

def generate_aug_params(m):
	base = torch.stack([torch.randperm(m, dtype = torch.float32) for i in range(4)], dim = 1)
	return (base + torch.rand(m, 4)) / m

def augment(input, opt, params = None):
	if opt.augment and params is not None:
		matrices = []
		for param in params:
			t = math.sqrt(param[0].item() * 2) - 1 if param[0].item() < 0.5 else 1 - math.sqrt((1 - param[0].item()) * 2)
			angle = t * opt.max_angle / 180 * math.pi
			scale = math.exp(param[1].item() * math.log(opt.max_scale))
			cmin = (opt.min_margin / scale) * 2 - 1
			cmax = (1 - opt.min_margin / scale) * 2 - 1
			cx = cmin + param[2].item() * (cmax - cmin)
			cy = cmin + param[3].item() * (cmax - cmin)
			matrices.append(torch.cat((torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) / scale, torch.tensor([[cx], [cy]])), dim = 1))
		grid = F.affine_grid(torch.stack(matrices, dim = 0), torch.Size((input.size(0), 4, input.size(2), input.size(3)))).to(input)
		return F.grid_sample(input, grid, mode = 'bilinear')
	else:
		return input
