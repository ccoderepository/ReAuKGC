import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
import time
import math
import copy
import os
import pickle
import numpy as np
import wandb
from conve_model import ConvE
import sys
from dataloader import MyDataset
save_folder = './params/'
from torch.utils.data import DataLoader


from torch.utils.data.distributed import DistributedSampler

class Trainer:
	def __init__(self, pre_data, model, tokenizer, optimizer, scheduler, device, hyperparams):

		self.pre_data = pre_data
		self.model = model

		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.device = device
		self.identifier = hyperparams['identifier']
		self.hyperparams = hyperparams
		self.save_folder = save_folder
		self.top_m = hyperparams['top_m']
		self.load_epoch = hyperparams['load_epoch']
		self.batch_size = hyperparams['batch_size']
		self.scheduler = scheduler
		# self.local_rank = hyperparams['local_rank']
		model.to(device)

		self.result_log = self.save_folder + self.identifier + '.txt'
		self.param_path_template = self.save_folder + self.identifier + '-epc_{0}_metric_{1}'  + '.pt'
		self.history_path = self.save_folder + self.identifier + '-history_{0}'  + '.pkl'
		self.conve = ConvE(pre_data).cuda()

		self.best_metric = {'acc': 0, 'f1': 0,
			'raw_mrr': 0, 'raw_hits1': 0, 'raw_hits3': 0, 'raw_hits10': 0,
			'fil_mr': 0, 'fil_mrr': 0, 'fil_hits1': 0, 'fil_hits3': 0, 'fil_hits10': 0,
		}

		self.best_epoch = {'acc': -1, 'f1': -1,
			'raw_mrr': -1, 'raw_hits1': -1, 'raw_hits3': -1, 'raw_hits10': -1,
			'fil_mr': -1, 'fil_mrr': -1, 'fil_hits1': -1, 'fil_hits3': -1, 'fil_hits10': -1,
		}

		self.history_value = {'acc': [], 'f1': [],
			'raw_mrr': [], 'raw_hits1': [], 'raw_hits3': [], 'raw_hits10': [],
			'fil_mr': [], 'fil_mrr': [], 'fil_hits1': [], 'fil_hits3': [], 'fil_hits10': [],
		}


		if not os.path.exists(save_folder):
			os.makedirs(save_folder)


		load_path = hyperparams['load_path']
		if load_path == None and self.load_epoch >= 0:
			load_path = self.param_path_template.format(self.load_epoch, hyperparams['load_metric'])
			history_path = self.history_path.format(self.load_epoch)
			if os.path.exists(history_path):
				with open(history_path, 'rb') as fil:
					self.history_value = pickle.load(fil)

		if load_path != None:
			if not (load_path.startswith(save_folder) or load_path.startswith('./saveparams/')):
				load_path = save_folder + load_path

			if os.path.exists(load_path):

				try:
					checkpoint = torch.load(load_path)
					model.load_state_dict(checkpoint['model_state_dict'], strict=False)

					optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
					scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

					print('Model & Optimizer  Parameters loaded from {0}.'.format(load_path))
				except:
					model.load_state_dict(torch.load(load_path), strict=False)
					print('Parameters loaded from {0}.'.format(load_path))
			else:
				print('Parameters {0} Not Found'.format(load_path))

		self.load_path = load_path


	def run(self):
		self.train()

	def train(self):
		model = self.model

		tokenizer = self.tokenizer
		optimizer = self.optimizer
		scheduler = self.scheduler

		device = self.device
		hyperparams = self.hyperparams

		batch_size = hyperparams['batch_size']
		epoch = hyperparams['epoch']


		pre_data = self.pre_data
		ent2id = pre_data.ent2id
		rel2id = pre_data.rel2id
		entity_list = pre_data.entity_list
		relation_list = pre_data.relation_list
		groundtruth = pre_data.get_groundtruth()


		conve_optimizer = torch.optim.Adam(self.conve.parameters(), lr=0.001, weight_decay=0.0)
		train_data = groundtruth['train']['tail']
		keys = [key for key in train_data.keys()]
		conve_batch_size = 128
		for conve_epoch in tqdm(range(50)):
			self.conve.train()
			index = [i for i in range(len(keys))]
			random.shuffle(index)
			for i in range(conve_batch_size, len(keys), conve_batch_size):
				conve_optimizer.zero_grad()
				iter_index = index[i - conve_batch_size: i]
				rel = torch.tensor([pre_data.rel2id[keys[m][0]] for m in iter_index]).cuda()
				sub = torch.tensor([pre_data.ent2id[keys[m][1]] for m in iter_index]).cuda()
				labels = []
				for iter in iter_index:
					labels.append([pre_data.ent2id[n] for n in train_data[keys[iter]]])
				y = torch.zeros([conve_batch_size, len(pre_data.ent2id)], dtype=torch.float32).cuda()
				for r in range(conve_batch_size):
					y[r, labels[r]] = 1.0
				y = (1.0 - 0.1) * y + (1.0 / len(pre_data.ent2id))
				pred = self.conve(sub, rel).squeeze()
				loss = self.conve.loss(pred, y)
				loss.backward()
				conve_optimizer.step()
			if (conve_epoch+1) % 10 == 0:
				self.conve.eval()
				with torch.no_grad():
					split = 'valid'
					ks = [1, 3, 10]
					MRR = {target: 0.0 for target in ['head', 'tail']}
					hits = {target: {k: 0.0 for k in ks} for target in ['head', 'tail']}

					for target in ['head', 'tail']:
						count_triples = 0
						for given in groundtruth[split][target].keys():
							expects = groundtruth[split][target][given]
							corrects = groundtruth['all'][given]

							rel = torch.tensor(pre_data.rel2id[given[0]]).cuda()
							sub = torch.tensor(pre_data.ent2id[given[1]]).cuda()
							pred = self.conve(sub, rel).squeeze()
							tops = pred.argsort(descending=True).tolist()
							for expect in expects:
								expect_id = pre_data.ent2id[expect]
								other_corrects = [correct for correct in corrects if correct != expect]
								other_correct_ids = set([pre_data.ent2id[c] for c in other_corrects])
								tops_ = [t for t in tops if (not t in other_correct_ids)]

								rank = tops_.index(expect_id) + 1
								MRR[target] += 1 / rank
								for k in ks:
									if rank <= k:
										hits[target][k] += 1
								count_triples += 1
						MRR[target] /= count_triples
						for k in ks:
							hits[target][k] /= count_triples
					hits1 = (hits['head'][1] + hits['tail'][1]) / 2
					hits3 = (hits['head'][3] + hits['tail'][3]) / 2
					hits10 = (hits['head'][10] + hits['tail'][10]) / 2
					total_MRR = (MRR['head'] + MRR['tail']) / 2
					print(hits1, hits3, hits10, total_MRR)


		# criterion
		criterion = torch.nn.CrossEntropyLoss()
		bce_criterion = torch.nn.BCELoss(reduction='none')
		sigmoid = torch.nn.Sigmoid()

		model.train()

		if hyperparams['wandb']:
			wandb.init(
				project="lmke",
				name=self.identifier,
				config=hyperparams
			)

		degrees = pre_data.statistics['degrees']

		best_score = 0
		data_sampler, n_batch = pre_data.get_type_train_dataset()
		dataset_size = len(data_sampler)
		real_dataset_size = dataset_size
		data = MyDataset(data_sampler)

		dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True,
								drop_last=False)
		early_stop = 0
		for epc in range(self.load_epoch+1, epoch):
			total_loss_link_prediction = 0
			time_begin = time.time()
			for i_b, batches in tqdm(enumerate(dataloader), total=n_batch):
				torch.cuda.empty_cache()  # 释放显存
				batch = []
				for i in (torch.arange(len(batches[0][0]))):
					temp = tuple([(batches[0][0][i], batches[0][1][i], batches[0][2][i]), batches[1][i]])
					batch.append(temp)
				triples = [i[0] for i in batch]
				triple_degrees = [ [degrees[e] for e in triple] for triple in triples]
				batch_size_ = len(batch)

				real_idxs = [ _ for _, i in enumerate(batch) if i[1] == 1]
				real_triples = [ i[0] for _, i in enumerate(batch) if i[1] == 1]
				real_triple_degrees = [ [degrees.get(e, 0) for e in triple] for triple in real_triples]

				real_batch_size = len(real_triples)

				real_inputs, real_positions = pre_data.batch_tokenize(real_triples)
				real_inputs.to(device)


				target_inputs, target_positions = pre_data.batch_tokenize_target(real_triples)
				target_inputs.to(device)

				label_idx_list = []

				labels = torch.zeros((len(real_triples), len(real_triples))).to(device)
				targets = [ triple[2] for triple in real_triples]
				target_idxs = [ ent2id[tar] for tar in targets]
				for i, triple in enumerate(real_triples):
					h, r, t = triple
					expects = set(groundtruth['train']['tail'][(r, h)])
					label_idx = [ i_t for i_t, target in enumerate(targets) if target in expects]
					label_idx_list.append(label_idx)
					labels[i, label_idx] = 1
				candidate_degrees = [ degrees.get(tar, 0) for tar in targets]

				loss = 0

				bce_loss = []
				preds_list = []


				target_preds = model(real_inputs, real_positions)
				target_encodes = model.encode_target(target_inputs, target_positions)
				preds = model.match(target_preds, target_encodes, real_triple_degrees)


				preds = sigmoid(preds)
				bce_loss.append(bce_criterion(preds, labels))
				preds_list.append(preds)

				for i in range(real_batch_size):

					pos_idx = sorted(label_idx_list[i])
					pos_set = set(pos_idx)
					neg_idx = [ _ for _ in range(labels.shape[1]) if not _ in pos_set]
					for j, bl in enumerate(bce_loss):
						# separately add lm_loss, transe_loss, and ensembled_loss
						l = bl[i]
						pos_loss = l[pos_idx].mean()

						# self-adversarial sampling
						neg_selfadv_weight = preds_list[j][i][neg_idx] #selfadv_weight[i][neg_idx]
						neg_weights = neg_selfadv_weight.softmax(dim=-1)
						neg_loss = (l[neg_idx]*neg_weights).sum()

						loss += pos_loss + neg_loss


				total_loss_link_prediction += loss.item()


				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				scheduler.step()


			avg_loss_link_prediction = total_loss_link_prediction / real_dataset_size

			time_end = time.time()
			time_epoch = time_end - time_begin
			print('Train: Epoch: {} ,   Avg_Link_Prediction_Loss: {}, Time: {}'.format(
				epc,  avg_loss_link_prediction, time_epoch))



			self.link_prediction(epc)
			if best_score < self.best_metric['fil_hits10']:
				self.link_prediction(epc, split='test')
				best_score = self.best_metric['fil_hits10']

		if hyperparams['wandb'] : wandb.finish()


	def link_prediction(self, epc=-1, split='valid'):
		model = self.model
		device = self.device
		hyperparams = self.hyperparams
		pre_data = self.pre_data
		top_m = self.top_m
		n_ent = model.n_ent
		n_rel = model.n_rel

		ent2id = pre_data.ent2id
		rel2id = pre_data.rel2id
		entity_list = pre_data.entity_list

		model.eval()

		sigmoid = torch.nn.Sigmoid()


		dataset = pre_data.get_dataset(split)
		groundtruth = pre_data.get_groundtruth()

		dl_statistics = pre_data.statistics
		degree_group = dl_statistics['degree_group']
		degrees = dl_statistics['degrees']


		ks = [1, 3, 10]
		MR = {
				setting:
					{target: 0 for target in ['head', 'tail']}
				for setting in ['raw', 'filter']}

		MRR = {
				setting:
					{target: 0 for target in ['head', 'tail']}
				for setting in ['raw', 'filter']
			 }


		hits = {
				setting:
					{target: {k: 0 for k in ks} for target in ['head', 'tail']}
				for setting in ['raw', 'filter']}





		# calc target embeddings
		batch_size = self.batch_size
		ent_target_encoded = torch.zeros((n_ent, model.hidden_size)).to(device)

		with torch.no_grad():
			# calc entity target embeddings
			random_map = [i for i in range(n_ent)]
			batch_list = [random_map[i:i+batch_size] for i in range(0, n_ent, batch_size)]

			for batch in batch_list:
				batch_targets = [ entity_list[_] for _ in batch]
				target_inputs, target_positions = pre_data.batch_tokenize_target(targets=batch_targets)
				target_inputs.to(device)
				target_encodes = model.encode_target(target_inputs, target_positions)
				ent_target_encoded[batch] = target_encodes

		f = open(self.result_log, 'a')
		print('Load Path: {} split: {}'.format(self.load_path, split), file=f)

		with torch.no_grad():
			self.conve.eval()
			for target in ['head', 'tail']:
				count_triples = 0
				total_triples = len(dataset)
				candidate_degrees = [degrees.get(tar, 0) for tar in entity_list]

				for given in tqdm(groundtruth[split][target].keys()):

					expects = groundtruth[split][target][given]
					corrects = groundtruth['all'][given]

					giv_rel = given[0]
					giv_ent = given[1]
					triples = [(giv_ent, giv_rel, expects[0])]

					triple_degrees = [ [ degrees.get(e, 0) for e in triple] for triple in triples]
					ent_list_degrees = [ degrees.get(e, 0) for e in entity_list]

					inputs, positions = pre_data.batch_tokenize(triples)
					inputs.to(device)

					target_preds = model(inputs, positions)
					target_encodes = ent_target_encoded

					preds = model.match(target_preds, target_encodes, triple_degrees, test=True, ent_list_degrees = ent_list_degrees)

					preds = sigmoid(preds)

					scores = preds.squeeze()
					scores = scores.argsort(descending=True).tolist()

					tops = scores[:top_m]
					rerank = self.conve.predict(torch.tensor(pre_data.ent2id[giv_ent]).cuda(), torch.tensor(pre_data.rel2id[giv_rel]).cuda(), torch.tensor(tops).cuda()).squeeze()
					rerank[:3] = rerank[:3] * 2.0
					rerank_dict = []
					for i in torch.arange(len(tops)):
						rerank_dict.append([tops[i], rerank[i]])
					rerank_result = sorted(rerank_dict, key=lambda x: x[1], reverse=True)
					scores[:top_m] = [result[0] for result in rerank_result]


					for expect in expects:
						expect_id = ent2id[expect]

						for setting in ['raw', 'filter']:
							if setting == 'raw':
								scores_ = scores
								rank = scores_.index(expect_id) + 1

							else:
								other_corrects = [correct for correct in corrects if correct != expect]
								other_correct_ids = set([ent2id[c] for c in other_corrects])
								scores_ = [ m for m in scores if (not m in other_correct_ids)]
								rank = scores_.index(expect_id) + 1
							MRR[setting][target] += 1/rank

							MR[setting][target] += rank

							for k in ks:
								if rank <= k:
									hits[setting][target][k] += 1

						count_triples += 1



				for setting in ['raw', 'filter']:
					MR[setting][target] /= count_triples
					MRR[setting][target] /= count_triples
					for k in ks:
						hits[setting][target][k] /= count_triples

					print('epc: {} {}: MR {:.5f} MRR {:.5f} hits 1 {:.5f} 3 {:.5f} 10 {:.5f}, Setting: {} Target: {} '.format(
						epc, split, MR[setting][target], MRR[setting][target], hits[setting][target][1], hits[setting][target][3], hits[setting][target][10],
						setting, target
					 ))


					print('epc: {} {}: MR {:.5f} MRR {:.5f} hits 1 {:.5f} 3 {:.5f} 10 {:.5f}, Setting: {} Target: {} '.format(
						epc, split, MR[setting][target], MRR[setting][target], hits[setting][target][1], hits[setting][target][3], hits[setting][target][10],
						setting, target
					 ), file=f)






		raw_mrr = (MRR['raw']['head'] + MRR['raw']['tail']) / 2
		raw_hits1 = (hits['raw']['head'][1] + hits['raw']['tail'][1]) / 2
		raw_hits3 = (hits['raw']['head'][3] + hits['raw']['tail'][3]) / 2
		raw_hits10 = (hits['raw']['head'][10] + hits['raw']['tail'][10]) / 2

		fil_mr = (MR['filter']['head'] + MR['filter']['tail']) / 2
		fil_mrr = (MRR['filter']['head'] + MRR['filter']['tail']) / 2
		fil_hits1 = (hits['filter']['head'][1] + hits['filter']['tail'][1]) / 2
		fil_hits3 = (hits['filter']['head'][3] + hits['filter']['tail'][3]) / 2
		fil_hits10 = (hits['filter']['head'][10] + hits['filter']['tail'][10]) / 2

		print('Overall:epc {} MR {:.5f} MRR {:.5f} hits 1 {:.5f} 3 {:.5f} 10 {:.5f}, Setting: Filter '.format(
			epc, fil_mr, fil_mrr, fil_hits1, fil_hits3, fil_hits10), file=f)
		print('Overall:epc {} MR {:.5f} MRR {:.5f} hits 1 {:.5f} 3 {:.5f} 10 {:.5f}, Setting: Filter '.format(
			epc, fil_mr, fil_mrr, fil_hits1, fil_hits3, fil_hits10))

		if split != 'test':
			self.update_metric(epc, 'raw_mrr', raw_mrr)
			self.update_metric(epc, 'raw_hits1', raw_hits1)
			self.update_metric(epc, 'raw_hits3', raw_hits3)
			self.update_metric(epc, 'raw_hits10', raw_hits10)

			self.update_metric(epc, 'fil_mr', fil_mr)
			self.update_metric(epc, 'fil_mrr', fil_mrr)
			# self.save_model(epc, 'fil_mrr', fil_mrr)
			self.update_metric(epc, 'fil_hits1', fil_hits1)
			# self.save_model(epc, 'fil_hits1', fil_hits1)
			self.update_metric(epc, 'fil_hits3', fil_hits3)
			self.update_metric(epc, 'fil_hits10', fil_hits10)
			# self.save_model(epc, 'fil_hits10', fil_hits10)

			if hyperparams['wandb']: wandb.log({'fil_mr': fil_mr, 'fil_mrr': fil_mrr, 'fil_hits1': fil_hits1, 'fil_hits3': fil_hits3, 'fil_hits10': fil_hits10, 'raw_hits10': raw_hits10})

		model.train()



	def update_metric(self, epc, name, score):
		self.history_value[name].append(score)
		if ( name not in ['fil_mr', 'raw_mr'] and score > self.best_metric[name]) or ( name in ['fil_mr', 'raw_mr'] and score < self.best_metric[name]):
			self.best_metric[name] = score
			self.best_epoch[name] = epc
			if name in ['fil_mr', 'raw_mr']:
				print('epc: {} !Metric {} Updated as: {:.2f}'.format(epc, name, score))
			else:
				print('epc: {}! Metric {} Updated as: {:.2f}'.format(epc, name, score*100))
			return True
		else:
			return False

	def save_model(self, epc, metric, metric_val):
		save_path = self.param_path_template.format(epc, metric)
		last_path = self.param_path_template.format(self.best_epoch[metric], metric)

		if self.update_metric(epc, metric, metric_val):
			if os.path.exists(last_path) and save_path != last_path and epc >= self.best_epoch[metric]:
				os.remove(last_path)
				print('Last parameters {} deleted'.format(last_path))

			#torch.save(self.model.state_dict(), save_path)
			torch.save({
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'scheduler_state_dict': self.scheduler.state_dict(),
			}, save_path)

			print('Parameters saved into ', save_path)


	def debug_signal_handler(self, signal, frame):
		pdb.set_trace()

	def log_best(self):
		print('Best Epoch {0} micro_f1 {1}'.format(self.best_epoch, self.best_metric))
