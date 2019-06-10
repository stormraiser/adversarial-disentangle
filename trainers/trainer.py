import os
import os.path
import pickle
import types

import torch

class Trainer:

	def __init__(self, options, subfolders = [], copy_keys = []):
		self.models = {}
		self.state = types.SimpleNamespace()
		self.logs = {}

		for key in copy_keys + ['device', 'save_path', 'load_path', 'niter']:
			self.__dict__[key] = options.__dict__[key]

		self.state.iter = 0
		self.funcs = []

		for subfolder in ['models', 'log'] + subfolders:
			if not os.path.exists(os.path.join(self.save_path, subfolder)):
				os.mkdir(os.path.join(self.save_path, subfolder))

		self.default_funcs = [
			(self.save_last, options.save_iter),
			(self.save_checkpoint, options.checkpoint_iter),
			(self.save_log, options.log_iter)
		]

	def add_model(self, name, model, optimizer):
		self.models[name] = (model, optimizer)

	def add_periodic_func(self, func, period):
		self.funcs.append((func, period))

	def save(self, prefix):
		for name, model_pair in self.models.items():
			model, optimizer = model_pair
			torch.save(model.state_dict(), os.path.join(self.save_path, 'models', '{0}_{1}.pt'.format(prefix, name)))
			torch.save(optimizer.state_dict(), os.path.join(self.save_path, 'models', '{0}_{1}_optim.pt'.format(prefix, name)))
		torch.save(self.state, os.path.join(self.save_path, 'models', '{0}_state.pt'.format(prefix)))

	def save_last(self):
		self.save('last')

	def save_checkpoint(self):
		self.save(self.state.iter)

	def load(self, prefix):
		for name, model_pair in self.models.items():
			model, optimizer = model_pair
			model.load_state_dict(torch.load(os.path.join(self.load_path, 'models', '{0}_{1}.pt'.format(prefix, name)), map_location = self.device))
			optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'models', '{0}_{1}_optim.pt'.format(prefix, name)), map_location = self.device))
		self.state = torch.load(os.path.join(self.load_path, 'models', '{0}_state.pt'.format(prefix)), map_location = self.device)

	def log(self, name, value):
		if name in self.logs:
			self.logs[name].append((self.state.iter, value))
		else:
			self.logs[name] = [(self.state.iter, value)]

	def save_log(self):
		with open(os.path.join(self.save_path, 'log', str(self.state.iter)), 'wb') as log_file:
			pickle.dump(self.logs, log_file)
		self.logs.clear()

	def train_one_iter(self):
		self.state.iter += 1
		self.iter_func()

		for func, period in self.funcs + self.default_funcs:
			if self.state.iter % period == 0:
				func()

	def iter_func(self):
		raise NotImplementedError

	def run(self):
		while self.state.iter < self.niter:
			self.train_one_iter()
