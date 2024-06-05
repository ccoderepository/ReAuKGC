import os
# import pdb
import random
import math
import pickle
import torch
import time
from tqdm import tqdm
import copy
from transformers import BatchEncoding
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_set):
        super(MyDataset, self).__init__()
        self.data_set = data_set  # 加载数据集
        self.length = len(data_set)  # 数据集长度

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data_set[index]


class preprocess_data(object):
    def __init__(self, in_paths, tokenizer, batch_size=16, max_desc_length=50, model='bert'):

        self.datasetName = in_paths['dataset']

        self.train_set = self.load_dataset(in_paths['train'])
        if self.datasetName not in ['fb13']:
            self.valid_set = self.load_dataset(in_paths['valid'])
            self.test_set = self.load_dataset(in_paths['test'])
            self.valid_set_with_neg = None
            self.test_set_with_neg = None
        else:
            self.valid_set, self.valid_set_with_neg = self.load_dataset_with_neg(in_paths['valid'])
            self.test_set, self.test_set_with_neg = self.load_dataset_with_neg(in_paths['test'])

        self.whole_set = self.train_set + self.valid_set + self.test_set

        self.uid2text = {}
        self.uid2tokens = {}

        self.entity_set = set([t[0] for t in (self.train_set + self.valid_set + self.test_set)] + [t[-1] for t in (
                    self.train_set + self.valid_set + self.test_set)])
        self.relation_set = set([t[1] for t in (self.train_set + self.valid_set + self.test_set)])

        self.tokenizer = tokenizer
        for p in in_paths['text']:
            self.load_text(p)

        self.batch_size = batch_size
        self.step_per_epc = math.ceil(len(self.train_set) / batch_size)

        self.train_entity_set = set([t[0] for t in self.train_set] + [t[-1] for t in self.train_set])
        self.train_relation_set = set([t[1] for t in self.train_set])

        self.entity_list = sorted(self.entity_set)
        self.relation_list = sorted(self.relation_set)

        self.ent2id = {e: i for i, e in enumerate(sorted(self.entity_set))}
        self.rel2id = {r: i for i, r in enumerate(sorted(self.relation_set))}

        self.id2ent = {i: e for i, e in enumerate(sorted(self.entity_set))}
        self.id2rel = {i: r for i, r in enumerate(sorted(self.relation_set))}

        self.max_desc_length = max_desc_length

        self.groundtruth, self.possible_entities = self.count_groundtruth()


        self.model = model

        self.orig_vocab_size = len(tokenizer)

        self.count_degrees()

    def load_dataset(self, in_path):
        dataset = []
        rev_rel_dataset = []
        with open(in_path, 'r', encoding='utf8') as fil:
            for line in fil.readlines():
                if in_path[-3:] == 'txt':
                    h, t, r = line.strip('\n').split('\t')
                else:
                    h, r, t = line.strip('\n').split('\t')
                dataset.append((h, r, t))
                rev_rel = 'reverse' + r
                rev_rel_dataset.append((t, rev_rel, h))
        dataset += rev_rel_dataset
        return dataset

    def load_dataset_with_neg(self, in_path):
        dataset = []
        dataset_with_neg = []
        with open(in_path, 'r', encoding='utf8') as fil:
            for line in fil.readlines():
                h, r, t, l = line.strip('\n').split('\t')

                if l == '-1':
                    l = 0
                else:
                    l = 1
                dataset.append((h, r, t))
                dataset_with_neg.append((h, r, t, l))
        return dataset, dataset_with_neg

    def get_type_train_dataset(self):

        dataset = []
        random.shuffle(self.train_set)
        for triple in tqdm(self.train_set):
            dataset.append((triple, 1))
        n_batch = math.ceil(len(dataset) / self.batch_size)
        return dataset, n_batch

    def load_text(self, in_path):
        uid2text = self.uid2text
        uid2tokens = self.uid2tokens

        tokenizer = self.tokenizer

        with open(in_path, 'r', encoding='utf8') as fil:
            for line in fil.readlines():
                uid, text = line.strip('\n').split('\t', 1)
                text = text.replace('@en', '').strip('"')
                if 'relation' not in in_path:
                    if uid not in uid2text.keys():
                        uid2text[uid] = text
                    tokens = tokenizer.tokenize(text)
                    if uid not in uid2tokens.keys():
                        uid2tokens[uid] = tokens
                else:
                    if uid not in uid2text.keys():
                        uid2text[uid] = text
                        uid2text['reverse' + uid] = 'reverse' + text
                    tokens = tokenizer.tokenize(text)
                    if uid not in uid2tokens.keys():
                        uid2tokens[uid] = tokens
                        uid2tokens['reverse' + uid] = ['reverse'] + tokens

        self.uid2text = uid2text
        self.uid2tokens = uid2tokens

    def triple_to_text(self, triple, with_text):

        tokenizer = self.tokenizer
        ent2id = self.ent2id
        rel2id = self.rel2id

        if True:
            # 512 tokens, 1 CLS, 1 SEP, 1 head, 1 rel, 1 tail, so 507 remaining.
            h_n_tokens = min(241, self.max_desc_length)
            t_n_tokens = min(241, self.max_desc_length)
            r_n_tokens = min(16, self.max_desc_length)

        h, r, t = triple

        h_text_tokens = self.uid2tokens.get(h, [])[:h_n_tokens] if with_text['h'] else []
        r_text_tokens = self.uid2tokens.get(r, [])[:r_n_tokens] if with_text['r'] else []
        t_text_tokens = self.uid2tokens.get(t, [])[:t_n_tokens] if with_text['t'] else []

        h_token = [self.tokenizer.cls_token] if with_text['h'] else [tokenizer.mask_token]
        r_token = [self.tokenizer.cls_token] if with_text['r'] else [tokenizer.mask_token]
        t_token = [self.tokenizer.cls_token] if with_text['t'] else [tokenizer.mask_token]

        tokens = h_token + h_text_tokens + r_token + r_text_tokens + t_token + t_text_tokens
        text = tokenizer.convert_tokens_to_string(tokens)

        return text, tokens

    def element_to_text(self, target):
        tokenizer = self.tokenizer
        ent2id = self.ent2id
        rel2id = self.rel2id

        n_tokens = min(508, self.max_desc_length)

        text_tokens = self.uid2tokens.get(target, [])[:n_tokens]

        token = [self.tokenizer.cls_token]  # '[CLS]']

        tokens = token + text_tokens

        text = tokenizer.convert_tokens_to_string(tokens)

        return text, tokens

    def get_ent2id(self):
        return self.ent2id

    def get_rel2id(self):
        return self.rel2id

    def batch_tokenize(self, batch_triples):
        batch_texts = []
        batch_tokens = []
        batch_positions = []

        ent2id = self.ent2id
        rel2id = self.rel2id
        sep_tokens = [self.tokenizer.sep_token]

        for triple in batch_triples:
            # text, tokens = self.triple_to_text(triple, with_text)
            h, r, t = triple
            r_text_tokens = self.uid2tokens.get(r, [])
            h_tokens = self.uid2tokens.get(h, [])
            tokens = h_tokens + sep_tokens + r_text_tokens
            # tokens = self.uid2tokens.get(h, [])
            text = self.tokenizer.convert_tokens_to_string(tokens)
            batch_texts.append(text)
            batch_tokens.append(tokens)

        # batch_tokens_ = self.tokenizer(batch_texts, truncation = True, max_length = 512, return_tensors='pt', padding=True )
        batch_tokens = self.my_tokenize(batch_tokens, max_length=self.max_desc_length, padding=True, model=self.model)

        orig_vocab_size = self.orig_vocab_size
        num_ent_rel_tokens = len(ent2id) + len(rel2id)

        mask_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        cls_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)

        for i, _ in enumerate(batch_tokens['input_ids']):
            triple = batch_triples[i]
            h, r, t = triple

            cls_pos = torch.where((_ == cls_idx))[0]

            batch_positions.append({'cls_tokens': (rel2id[r], cls_pos.item())})

        return batch_tokens, batch_positions

    def batch_tokenize_target(self, batch_triples=None, targets=None):
        batch_texts = []
        batch_tokens = []
        batch_positions = []

        ent2id = self.ent2id
        rel2id = self.rel2id

        if targets == None:
            targets = [triple[2] for triple in batch_triples]

        for target in targets:
            # text, tokens = self.element_to_text(target)
            tokens = self.uid2tokens.get(target, [])
            text = self.tokenizer.convert_tokens_to_string(tokens)
            batch_texts.append(text)
            batch_tokens.append(tokens)

        batch_tokens = self.my_tokenize(batch_tokens, max_length=self.max_desc_length, padding=True, model=self.model)

        for i, _ in enumerate(batch_tokens['input_ids']):
            target = targets[i]
            target_pos = 0

            if target in ent2id.keys():
                target_idx = ent2id[target]
            else:
                target_idx = rel2id[target]

            batch_positions.append((target_idx, target_pos))

        return batch_tokens, batch_positions

    def get_dataset_size(self, split='train'):
        if split == 'train':
            return len(self.train_set)

    def count_groundtruth(self):
        groundtruth = {split: {'head': {}, 'tail': {}} for split in ['train', 'valid', 'test']}
        groundtruth['all'] = {}
        possible_entities = {split: {'head': {}, 'tail': {}} for split in ['train']}

        for triple in self.train_set:
            h, r, t = triple
            groundtruth['all'].setdefault((r, h), [])
            groundtruth['all'][(r, h)].append(t)
            groundtruth['train']['tail'].setdefault((r, h), [])
            groundtruth['train']['tail'][(r, h)].append(t)
            possible_entities['train']['tail'].setdefault(r, set())
            possible_entities['train']['tail'][r].add(t)

        for triple in self.valid_set:
            h, r, t = triple
            groundtruth['all'].setdefault((r, h), [])
            groundtruth['all'][(r, h)].append(t)
            if 'reverse' in r:
                groundtruth['valid']['head'].setdefault((r, h), [])
                groundtruth['valid']['head'][(r, h)].append(t)
            else:
                groundtruth['valid']['tail'].setdefault((r, h), [])
                groundtruth['valid']['tail'][(r, h)].append(t)

        for triple in self.test_set:
            h, r, t = triple
            groundtruth['all'].setdefault((r, h), [])
            groundtruth['all'][(r, h)].append(t)
            if 'reverse' in r:
                groundtruth['test']['head'].setdefault((r, h), [])
                groundtruth['test']['head'][(r, h)].append(t)
            else:
                groundtruth['test']['tail'].setdefault((r, h), [])
                groundtruth['test']['tail'][(r, h)].append(t)

        return groundtruth, possible_entities

    def get_groundtruth(self):
        return self.groundtruth

    def get_dataset(self, split):
        assert (split in ['train', 'valid', 'test'])

        if split == 'train':
            return self.train_set
        elif split == 'valid':
            return self.valid_set
        elif split == 'test':
            return self.test_set

    def my_tokenize(self, batch_tokens, max_length=50, padding=True, model='roberta'):
        '''
        if model == 'roberta':
            start_tokens = ['<s>']
            end_tokens = ['</s>']
            pad_token = '<pad>'
        elif model == 'bert':
            start_tokens = ['[CLS]']
            end_tokens = ['[SEP]']
            pad_token = '[PAD]'
        '''

        start_tokens = [self.tokenizer.cls_token]
        end_tokens = [self.tokenizer.sep_token]

        batch_tokens = [start_tokens + i + end_tokens for i in batch_tokens]

        batch_size = len(batch_tokens)
        longest = min(max([len(i) for i in batch_tokens]), self.max_desc_length)

        if model == 'bert':
            input_ids = torch.zeros((batch_size, longest)).long()
        elif model == 'roberta':
            input_ids = torch.ones((batch_size, longest)).long()

        token_type_ids = torch.zeros((batch_size, longest)).long()
        attention_mask = torch.zeros((batch_size, longest)).long()

        for i in range(batch_size):
            index_entity = batch_tokens[i].index('[SEP]')
            tokens = self.tokenizer.convert_tokens_to_ids(batch_tokens[i])
            tokens = tokens[0: min(max([len(i) for i in batch_tokens]), 50)]
            input_ids[i, :len(tokens)] = torch.tensor(tokens).long()
            attention_mask[i, :len(tokens)] = 1
            token_type_ids[i, index_entity + 1:len(tokens)] = 1

        if model == 'roberta':
            return BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask})
        elif model == 'bert':
            return BatchEncoding(
                data={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})

    def count_degrees(self):
        train_set = self.train_set  # + self.valid_set + self.test_set
        degrees = {}

        for triple in train_set:
            h, r, t = triple
            degrees[h] = degrees.get(h, 0) + 1
            degrees[t] = degrees.get(t, 0) + 1
            degrees[r] = degrees.get(r, 0) + 1

        for k, v in degrees.items():
            degrees[k] = v / 2

        raw_degrees = copy.deepcopy(degrees)

        max_degree = 0
        for k, v in degrees.items():
            max_degree = max(max_degree, v)
        max_degree = math.floor(math.log(max_degree) / math.log(2))
        count_degree_group = {i: 0 for i in range(0, max_degree + 1)}

        for k, v in degrees.items():
            degrees[k] = math.floor(math.log(v) / math.log(2)) + 1
            count_degree_group[degrees[k]] = count_degree_group.get(degrees[k], 0) + 1

        self.statistics = {
            'degrees': raw_degrees,
            'degree_group': degrees,
            'count_degree_group': count_degree_group,
            'max_degree': max_degree
        }



