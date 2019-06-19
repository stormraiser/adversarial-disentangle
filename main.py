import argparse
from options import process_options
import trainers
from encode import FullDatasetEncoder

parser = argparse.ArgumentParser()

# master options
parser.add_argument('cmd',
	choices = ['stage1', 'encode', 'classifier', 'stage2', 'gan'],
	help = 'action to be performed')
parser.add_argument('--load_path',
	help = 'load a saved experiment')
parser.add_argument('--load_iter',
	help = 'iteration of saved model to load')
parser.add_argument('--save_path',
	help = 'path for saving the experiment')

# loading previous stages in later stages
parser.add_argument('--enc_path',
	help = 'path to saved encoder')
parser.add_argument('--enc_iter',           type = int,
	help = 'iteration of saved encoder to load')
parser.add_argument('--cla_path',
	help = 'path to saved classifier')
parser.add_argument('--cla_iter',           type = int,
	help = 'iteration of saved classifier to load')

parser.add_argument('--reset_gen',                        action = 'store_true',
	help = 'do not inherit stage 1 generator')
parser.add_argument('--reset_sty',                        action = 'store_true',
	help = 'do not inherit of stage 1 style codes')
parser.add_argument('--reset_dis',                        action = 'store_true',
	help = 'do not initialize discriminator with pre-trained c2')
parser.add_argument('--reset_cla',                        action = 'store_true',
	help = 'do not inherit pre-trained c2')

# dataset
parser.add_argument('--data_root',
	help = 'path to dataset')
parser.add_argument('--weight_root',
	help = 'path to reconstruction weight maps')
parser.add_argument('--image_size',         type = int,
	help = 'size of image')
parser.add_argument('--crop_size',          type = int,
	help = 'center crop size before resizing')

# base network structures
parser.add_argument('--features',           type = int,   nargs = '+',
	help = 'number of features')
parser.add_argument('--blocks',                           nargs = '+',
	help = 'sequence of blocks')
parser.add_argument('--adain_features',     type = int,
	help = 'number of AdaIn features')
parser.add_argument('--adain_blocks',       type = int,
	help = 'number of AdaIn blocks')

parser.add_argument('--content_size',       type = int,
	help = 'length of content code')
parser.add_argument('--style_size',         type = int,
	help = 'length of style code')

# network structure for each network, overrides the base
parser.add_argument('--enc_features',       type = int,   nargs = '+',
	help = 'number of features in encoder')
parser.add_argument('--enc_blocks',                       nargs = '+',
	help = 'sequence of blocks in encoder')
parser.add_argument('--enc_adain_features', type = int,
	help = 'number of AdaIn features in encoder')
parser.add_argument('--enc_adain_blocks',   type = int,
	help = 'number of AdaIn blocks in encoder')

parser.add_argument('--gen_features',       type = int,   nargs = '+',
	help = 'number of features in generator')
parser.add_argument('--gen_blocks',                       nargs = '+',
	help = 'sequence of blocks in generator')
parser.add_argument('--gen_adain_features', type = int,
	help = 'number of AdaIn features in generator')
parser.add_argument('--gen_adain_blocks',   type = int,
	help = 'number of AdaIn blocks in generator')

parser.add_argument('--cla_features',       type = int,   nargs = '+',
	help = 'number of features in classifier')
parser.add_argument('--cla_blocks',                       nargs = '+',
	help = 'sequence of blocks in classifier')
parser.add_argument('--cla_adain_features', type = int,
	help = 'number of AdaIn features in classifier')
parser.add_argument('--cla_adain_blocks',   type = int,
	help = 'number of AdaIn blocks in classifier')

parser.add_argument('--dis_features',       type = int,   nargs = '+',
	help = 'number of features in discriminator')
parser.add_argument('--dis_blocks',                       nargs = '+',
	help = 'sequence of blocks in discriminator')
parser.add_argument('--dis_adain_features', type = int,
	help = 'number of AdaIn features in discriminator')
parser.add_argument('--dis_adain_blocks',   type = int,
	help = 'number of AdaIn blocks in discriminator')

parser.add_argument('--mlp',                              action = 'store_true',
	help = 'use MLP in stage 1')
parser.add_argument('--mlp_features',       type = int,
	help = 'number of features in MLP classifier')
parser.add_argument('--mlp_layers',         type = int,
	help = 'number of layers in MLP classifier')

# weighting
parser.add_argument('--con_weight',         type = float,
	help = 'weight of content loss')
parser.add_argument('--sty_weight',         type = float,
	help = 'weight of style loss')
parser.add_argument('--cla_weight',         type = float,
	help = "weight of generator's loss against classifier")
parser.add_argument('--dis_weight',         type = float,
	help = "weight of generator's loss against discriminator")
parser.add_argument('--rec_weight',         type = float,
	help = 'weight of reconstruction loss')

parser.add_argument('--match_weight',       type = float,
	help = 'weight of content matching loss')

parser.add_argument('--cla_fake',           choices = ['true', 'false'],
	help = "add NLU of fake samples to classifier's loss")

parser.add_argument('--dis_br_weight',      type = float,
	help = "weight of discriminator's batch regularization loss")
parser.add_argument('--cla_br_weight',      type = float,
	help = "weight of classifier's batch regularization loss")
parser.add_argument('--gen_br_weight',      type = float,
	help = "weight of generator's batch regularization loss")
parser.add_argument('--transform_penalty',  type = float,
	help = "weight of penalty of bad spatial transformation grid")

#training
parser.add_argument('--lr',                 type = float,
	help = 'learning rate')
parser.add_argument('--sty_lr',             type = float,
	help = 'learning rate of style codes')
parser.add_argument('--lr_ramp',            type = int,
	help = 'ramp up lr from 0 in the first few iterations')

parser.add_argument('--force_dis',          type = float,
	help = 'keep training discriminator until its loss is lower than threshold')
parser.add_argument('--force_cla',          type = float,
	help = 'keep training classifier until its loss is lower than threshold')

parser.add_argument('--augment',            choices = ['true', 'false'],
	help = 'enable discriminator and classifier augmentation')
parser.add_argument('--max_angle',          type = float,
	help = 'maximum rotation angle')
parser.add_argument('--max_scale',          type = float,
	help = 'maximum scale')
parser.add_argument('--min_margin',         type = float,
	help = 'minimum distance of center of crop to image border')

parser.add_argument('--content_dropout',    type = float,
	help = 'base of exponential nested dropout of content code')
parser.add_argument('--style_dropout',      type = float,
	help = 'base of exponential nested dropout of style code')

parser.add_argument('--batch_size',         type = int,
	help = 'batch size')
parser.add_argument('--nloader',            type = int,
	help = 'number of loaders')

parser.add_argument('--niter',              type = int,
	help = 'number of iterations')
parser.add_argument('--save_iter',          type = int,
	help = 'interval of saving the latest model')
parser.add_argument('--checkpoint_iter',    type = int,
	help = 'interval of saving a separate checkpoint')
parser.add_argument('--log_iter',           type = int,
	help = 'interval of saving logs')

parser.add_argument('--visualize_iter',     type = int,
	help = 'interval of saving generated samples')
parser.add_argument('--visualize_size',     type = int,   nargs = 2,
	help = 'size of visualization grid')

parser.add_argument('--device',
	help = 'device (torch device string)')

options = process_options(parser.parse_args())

stage_classes = {
	'stage1' : trainers.Stage1Trainer,
	'stage2' : trainers.Stage2Trainer,
	'classifier' : trainers.ClassifierTrainer,
	'gan' : trainers.GANTrainer,
	'encode' : FullDatasetEncoder
}

stage_classes[options.cmd](options).run()
