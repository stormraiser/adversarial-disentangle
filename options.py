import types
import os.path
import json

from datasets import ImageFolder
import torchvision.transforms as T

defaults = {
	'lr_ramp'           : 100,
	'transform_penalty' : 100,
	'dis_br_weight'     : 1,
	'gen_br_weight'     : 1,
	'cla_br_weight'     : 1,
	'augment'           : 'true',
	'max_angle'         : 15,
	'max_scale'         : 1.25,
	'min_margin'        : 0.4,
	'nloader'           : 1,
	'save_iter'         : 1000,
	'log_iter'          : 1000,
	'checkpoint_iter'   : 20000,
	'visualize_iter'    : 100,
	'device'            : 'cuda:0',
	'adain_features'    : -1,
	'adain_blocks'      : -1,
	'mlp_layers'        : 10,
	'visualize_size'    : (None, None),
	'force_dis'         : 0.8,
	'force_cla'         : 0.6931,
	'niter'             : 1000000
}
stage_defaults = {
	'stage1' : {
		'lr'                : 5e-5,
		'sty_lr'            : 0.005,
		'rec_weight'        : 1,
		'con_weight'        : 1e-5,
		'sty_weight'        : 1e-5,
		'cla_weight'        : 0.2,
		'match_weight'      : 0,
		'cla_br_weight'     : 0
	},
	'encode' : {},
	'classifier' : {
		'lr'                : 1e-4
	},
	'stage2' : {
		'lr'                : 2e-5,
		'sty_lr'            : 0.002,
		'con_weight'        : 0.1,
		'sty_weight'        : 1e-5,
		'cla_weight'        : 1,
		'dis_weight'        : 1,
		'cla_fake'          : 'true'
	},
	'gan' : {
		'lr'                : 2e-5
	}
}

net_keys = ['features', 'blocks', 'adain_features', 'adain_blocks']
enc_keys = ['enc_features', 'enc_blocks', 'enc_adain_features', 'enc_adain_blocks']
gen_keys = ['gen_features', 'gen_blocks', 'gen_adain_features', 'gen_adain_blocks']
dis_keys = ['dis_features', 'dis_blocks', 'dis_adain_features', 'dis_adain_blocks']
cla_keys = ['cla_features', 'cla_blocks', 'cla_adain_features', 'cla_adain_blocks']
mlp_keys = ['mlp', 'mlp_features', 'mlp_layers']
augment_keys = ['augment', 'max_angle', 'max_scale', 'min_margin']

load_keys = enc_keys + gen_keys + dis_keys + cla_keys + mlp_keys\
	+ ['enc_path', 'enc_iter', 'content_size', 'style_size', 'image_size', 'vis_row', 'vis_col', 'nclass']
override_keys = augment_keys + ['data_root', 'weight_root', 'crop_size', 'con_weight', 'match_weight',
	'sty_weight', 'cla_weight', 'dis_weight', 'rec_weight', 'dis_br_weight',
	'cla_br_weight', 'gen_br_weight', 'transform_penalty', 'lr', 'sty_lr', 'lr_ramp',
	'force_dis', 'force_cla',
	'content_dropout', 'style_dropout', 'batch_size', 'nloader', 'niter',
	'save_iter', 'checkpoint_iter', 'log_iter', 'visualize_iter', 'device', 'cla_fake']

default_sizes = [
	# image_min  image_max                  features                 content  batch vis_col
	(        12,        20,             (128, 256, 512),                  32,    64,     10),
	(        22,        40,         (64, 128, 256, 512),                  64,    64,      8),
	(        42,        80,         (64, 128, 256, 512, 1024),           128,    32,      6),
	(        82,       160,         (64, 128, 256, 512, 1024, 2048),     256,    16,      4),
	(       162,       320,     (32, 64, 128, 256, 512, 1024, 2048),     512,     8,      3),
	(       322,       640, (16, 32, 64, 128, 256, 512, 1024, 2048),     512,     4,      2)
]

def process_options(options):
	if options.cmd in ['encode', 'stage2']:
		if options.enc_path is None:
			raise ValueError('encoder path must be specified')
	if options.cmd in ['stage1', 'classifier', 'stage2', 'gan']:
		options.save_path = options.save_path or options.load_path
		if options.save_path is None:
			raise ValueError('save path must be specified')

	if options.augment is not None:
		options.augment = options.augment == 'true'
	if options.cla_fake is not None:
		options.cla_fake = options.cla_fake == 'true'

	if options.load_path is not None:
		with open(os.path.join(options.load_path, 'options')) as file:
			saved_options = json.load(file)
		for key in load_keys:
			options.__dict__[key] = saved_options[key]
		for key in override_keys:
			if options.__dict__[key] is None:
				options.__dict__[key] = saved_options[key]
		options.load_iter = options.load_iter or 'last'
	else:
		if options.cmd in ['encode', 'stage2']:
			with open(os.path.join(options.enc_path, 'options')) as file:
				enc_options = json.load(file)

			print('using encoder structure from stage 1')
			for key in enc_keys + ['data_root', 'image_size', 'content_size']:
				options.__dict__[key] = enc_options[key]

			if options.cmd == 'stage2':
				if not options.reset_gen:
					print('using generator structure from stage 1')
					for key in gen_keys:
						options.__dict__[key] = enc_options[key]

				if not (options.reset_gen and options.reset_sty):
					print('using length of style code from stage 1')
					options.style_size = enc_options['style_size']

		if options.cmd == 'stage2' and options.cla_path is not None:
			with open(os.path.join(options.cla_path, 'options')) as file:
				cla_options = json.load(file)

			if cla_options['image_size'] != options.image_size:
				raise ValueError('image size of stage 1 networks and classifier 2 does not match')

			if not options.reset_dis:
				print('using discriminator structure from pre-trained classifier')
				for key1, key2 in zip(dis_keys, cla_keys):
					options.__dict__[key1] = cla_options[key2]

			if not options.reset_cla:
				print('using classifier structure from pre-trained classifier')
				for key in cla_keys:
					options.__dict__[key] = cla_options[key]

		for key, value in stage_defaults[options.cmd].items():
			if options.__dict__[key] is None:
				options.__dict__[key] = value
		for key, value in defaults.items():
			if options.__dict__[key] is None:
				options.__dict__[key] = value

		dataset = ImageFolder(options.data_root, transform = T.ToTensor())
		if options.image_size is None:
			if options.crop_size is None:
				print('image size not specified, using image size of dataset')
				options.image_size = dataset[0][0].size(1)
			else:
				print('image size not specified, using crop size')
				options.image_size = options.crop_size

		options.nclass = dataset.get_nclass()
		if options.style_size is None:
			print('style size not specified, using defaults')
			options.style_size = min(max(min(options.nclass // 4, 256), 16), options.nclass)

		if options.image_size % 2 != 0:
			raise ValueError('image size must be an even integer')

		options.vis_row = options.visualize_size[0]
		options.vis_col = options.visualize_size[1]

		for min_size, max_size, features, content_size, batch_size, vis_col in default_sizes:
			if min_size <= options.image_size <= max_size:
				options.features = options.features or features
				options.mlp_features = options.mlp_features or features[-1]
				if options.blocks is None:
					options.blocks = (['cc'] + ['cbc'] * (len(options.features) - 2) + ['f'])
					options.gen_blocks = options.gen_blocks or (['tc'] + ['cbc'] * (len(options.features) - 2) + ['f'])
				options.content_size = options.content_size or content_size
				options.batch_size = options.batch_size or (batch_size * 2 if options.cmd in ['classifier', 'gan', 'encode'] else batch_size)
				options.vis_col = options.vis_col or (vis_col * 2 if options.cmd in ['gan'] else vis_col)
		if options.content_size is None:
			raise ValueError('content size not specified and failed to set defaults')
		if options.batch_size is None:
			raise ValueError('batch size not specified and failed to set defaults')
		options.vis_col = options.vis_col or (10 if options.image_size < 16 else 2)
		options.vis_row = options.vis_row or (options.vis_col if options.cmd in ['gan'] else options.vis_col * 2)

		options.content_dropout = options.content_dropout or (1 - 2 / options.content_size)
		options.style_dropout = options.style_dropout or (1 - 2 / options.style_size)

		if options.cmd == 'stage1':
			for key1, key2 in zip(enc_keys, net_keys):
				if options.__dict__[key1] is None:
					options.__dict__[key1] = options.__dict__[key2]
				if options.__dict__[key1] is None:
					raise ValueError('encoder structure incomplete and failed to set defaults')

		if options.cmd in ['stage1', 'stage2', 'gan']:
			for key1, key2 in zip(gen_keys, net_keys):
				if options.__dict__[key1] is None:
					options.__dict__[key1] = options.__dict__[key2]
				if options.__dict__[key1] is None:
					raise ValueError('generator structure incomplete and failed to set defaults')

		if options.cmd == 'classifier' or (options.cmd == 'stage1' and not options.mlp):
			for key1, key2 in zip(cla_keys, net_keys):
				if options.__dict__[key1] is None:
					options.__dict__[key1] = options.__dict__[key2]
				if options.__dict__[key1] is None:
					raise ValueError('classifier structure incomplete and failed to set defaults')

		if options.cmd in ['stage2', 'gan']:
			for key1, key2 in zip(dis_keys, net_keys):
				if options.__dict__[key1] is None:
					options.__dict__[key1] = options.__dict__[key2]
				if options.__dict__[key1] is None:
					raise ValueError('discriminator structure incomplete and failed to set defaults')

	if options.cmd in ['stage1', 'classifier', 'stage2', 'gan']:
		save_options = {}
		for key in load_keys + override_keys:
			save_options[key] = options.__dict__[key]
		if not os.path.exists(options.save_path):
			os.makedirs(options.save_path)
		with open(os.path.join(options.save_path, 'options'), 'w') as file:
			json.dump(save_options, file)

		options.augment_options = types.SimpleNamespace()
		for key in augment_keys:
			options.augment_options.__dict__[key] = options.__dict__[key]

	return options
