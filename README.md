
Official python 3 PyTorch implementation of the paper [Disentangling Style and Content in Anime Illustrations](https://arxiv.org/abs/1905.10742).

The current version of the code is written from scratch after finishing the paper as the original code used for the experiments was the result of 2 years of evolution and became rather messy. I've not tested everything thoroughly and the documentation is still a work in progress but the default training options should work.

# Usage

Since it might be useful to e.g. experiment with different stage 2 configurations using the same stage 1 encoder I decided that it's best to train each stage separately rather than finishing the whole training in one run.

## Stage 1
To train stage 1 with minimal amount of configuration and default for everything else:
```
python main.py stage1 --data_root /path/to/dataset --save_path /path/to/save/folder
```
You may also want to set `--weight_root` and `--match_weight` which are not set by default but could be helpful. See below for explanations.

Options:
`--lr`, `--batch_size`: works as you would expect.

`--sty_lr`: separate option for learning rate of style codes.

`--nloader`: number of data loader processes.

`--device`: PyTorch device string, e.g. `cuda:0`.

`--niter`: training time, in iterations. Not sure if this is useful at all since I always just set a big number and ctrl-c when the result is good.

`--lr_ramp`: sometimes for the first few iterations the rms of gradient for the adaptive optimization algorithms is not accurate enough and causes bad weight updates. Ramp up learning rate from 0 in the beginning to avoid this. Shouldn't need to change this setting.

`--weight_root`: path to per-pixel weight map for reconstruction loss if you have them. I added this to try to artificially increase the weight of small but visually import features (eyes and mouth) to make them more consistent. Example weight map is provided in the dataset.

`--image_size`: size of input/output image. Training images will be resized to this size. Does not have to be power of 2 but must be an even number. If not specified, uses the height of the first (according to PyTorch's `ImageFolder`) image in the dataset. This will also determine the default network structure, batch size, size of content code and size of visualization grid if those are not specified.

`--crop_size`: if specified, crops training images from center before resizing.

`--features`: list of numbers of output channels in each level of the network. Each level consists of everything from one stride-2 block to before the next stride-2 block. The number of levels will be the length of the list minus 1, and the number of channels in each layer will be the corresponding entry in the list. The last number of the list is for the number of features of the fully connected layers.

`--blocks`: list of strings representing the sequence of blocks in each level. I'm experimenting with layers other than convolutions but those are not documented yet so for now just ignore this and use the defaults. Or read the code and see what other stuff are available.

`--adain_features`, `--adain_blocks`: set these to add adaptive instance normalization. Still needs some testing.

`--content_size`, `--style_size`: length of content and style codes.

`--rec_weight`, `--cla_weight`, `--con_weight`, `--sty_weight`: weight of reconstruction/adversarial loss against classifier/content code KL-divergence loss/style code KL-divergence loss.

`--match_weight`: in stage 1, add ||E(G(E(x), S(a')))-E(G(E(x), s(a)))||_2^2 to the generator. This is added after writing the paper and disabled by default. Helps with consistency of content with different styles but also requires more training time. You may want to try setting this to around 0.001.

`--augment`: true/false. Enable data augmentation for the classifier.

`--max_angle`, `--max_scale` maximum rotation angle (in degrees) and up-sampling scale for augmentation.

`--content_dropout`, `--style_dropout`: set to apply nested dropout to content/style code. Basically, with nested dropout, you keep a prefix of random length and set the suffix to zero. Set these parameters to p will cause the prefix of length k to be kept with probability p^(k-1). Nested dropout is enabled by default. Set these to 1 to disable.

`--save_iter`: interval to save the latest models. These will be saved with a prefix "last" and a new save will overwrite the old one.

`--checkpoint_iter`: interval to save separate checkpoints. These will be saved with a prefix being the current iteration so new saves will not overwrite old ones.

`--visualize_iter`: interval to save generated samples.

`--visualize_size`: number of rows and columns of the grid of visualization inputs. For each visualization input, three samples are generated: reconstruction with the style of the correct artist, with zero style, and with the style of the artist of the next visualization input.

`--load_path` set this to load and continue a previous training. The training options are saved along with the models so if you load a previous training there is no need to set any other options. But if you do the new values will override the saved values.

## Prepare Content Code
Since in stage 2 the encoder is fixed, we don't need to compute E(x) every time we want to use it. We can simply compute and save them before training stage 2. To do this:
```
python main.py encode --enc_path /path/to/stage1/folder --enc_iter iteration_to_load
```
`enc_iter` must be a numbered checkpoint and can't be `last`. This will encode the full dataset and also compute some statistics.

## Pre-train Stage 2 Classifier
```
python main.py classifier --data_root /path/to/dataset --save_path /path/to/save/folder
```

## Stage 2
```
python main.py stage2 --enc_path /path/to/stage1/folder --enc_iter iteration_to_load --cla_path /path/to/classifier/pretrain/folder --cla_iter iteration_to_load --save_path /path/to/save/folder
```

Some more options:
`--reset_gen`, `--reset_sty`, `--reset_dis`, `--reset_cla`: if any of these flags are set, the respective networks will not be initialized with the same network in previous stages.

`--dis_weight`: weight of generator's adversarial loss against discriminator.

`--con_weight`: in stage 2 this is the weight of content loss.

`--cla_fake`: true/false. If false, will not compute classifier's loss on generated samples. Defaults to true.

`--dis_br_weight`, `--cla_br_weight`, `--gen_br_weight`: something I call "Batch Regularization". It's complicated. Roughly, add a penalty to encourage the discriminator/classifier to produce nice statistics and the generator to generate fake samples that matches the statistics of the real samples. Just leave them there or set to 0 and see what may happen.

`--force_dis`: if larger than 0, will in each iteration train the discriminator with however many batches necessary until the running average loss of the past several batches is below this value, before training the generator. Using this and batch regularization together could cause several hundred batches to be used in the first few iterations. But don't be scared as very soon the discriminator will become very strong and will rarely need more than 1 batch.

`--force_cla`: same thing for the classifier.

# Dataset

The dataset used for the paper can be downloaded from [here](https://drive.google.com/file/d/130D159sNTpYFwjlGdnhxHkysMfqN-RUx/view?usp=sharing). I's a bit old, I gathered it in 2017, and the selecting process has been conservative: I removed images with too much occlusion including those wearing big hats since that makes it even more difficult to generate coherent hair that are cleanly separated from the background. I also removed badly localized/oriented detections instead of attempting to manually fix them, and there was quite some of them since the detector I've been using was not so good. Then I removed NSFW images, the very explicit ones at least, so that nothing weird accidentally makes it into the paper. The result is that I had one order of magnitude fewer images in the final dataset than what could potentially be available. I think I'll build some face detection/alignment tools and make a better dataset if I have time.

The training images were in PNG format, the major concern being that many face patches are cropped from near the border in the original image and contain missing pixels, and they should not simply be replaced by a uniform black or white background since black/white pixels and missing pixels should be different. So I marked the missing pixels with zero alpha value. The code is written to handle missing pixels in the training images correctly. Sadly the PNG images are too large to share, so to reduce size I had to split them into RGB part and alpha part with RGB part in JPG format. Script is provided to recombine them.

The weight map is generated semi-automatically. It's not very accurate but should serve the purpose.

In principle the same method can be used on any dataset labelled by classes, which means pretty much any dataset. These class do not have to correspond to different styles since the method does not contain any element specifically designed to deal with style vs. content. It only uses the class label without knowing what the label means. That being said, it is not clear how one should interpret the result on a general classification dataset. I've not done such experiments yet and I'm curious to see the results.