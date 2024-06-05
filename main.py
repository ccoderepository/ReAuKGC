# coding=UTF-8
import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


from trainer import Trainer
from dataloader import preprocess_data
from model import ReAuKGC

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--model_lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--data', type=str, default='wn18rr')
    parser.add_argument('--plm', type=str, default='bert',
                        choices=['bert', 'bert_tiny', 'deberta', 'deberta_large', 'roberta', 'roberta_large'])
    parser.add_argument('--description', type=str, default='desc')

    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=-1)
    parser.add_argument('--load_metric', type=str, default='hits1')

    parser.add_argument('--max_desc_length', type=int, default=50)

    # directly run test
    parser.add_argument('--link_prediction', default=False, action='store_true')
    parser.add_argument('--prefix_tuning', default=False, action='store_true',
                        help='fix language models and only tune added components')

    parser.add_argument('--self_adversarial', default=True, help='self adversarial negative sampling')
    parser.add_argument('--top_m', default=6, type=int, help='the top m that need to be re-ranking')

    parser.add_argument('--wandb', default=False, action='store_true')


    arg = parser.parse_args()

    identifier = '{}-{}-{}-batch_size={}-prefix_tuning={}-max_desc_length={}'.format(arg.data, arg.plm, arg.description,
                                                                                     arg.batch_size, arg.prefix_tuning,
                                                                                     arg.max_desc_length)
    # Set random seed
    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)

    device = torch.device('cuda')

    if arg.plm == 'bert':
        plm_name = "bert-base-uncased"
        t_model = 'bert'
    elif arg.plm == 'bert_tiny':
        plm_name = "prajjwal1/bert-tiny"
        t_model = 'bert'
    elif arg.plm == 'deberta':
        plm_name = 'microsoft/deberta-v3-base'
        t_model = 'bert'
    elif arg.plm == 'deberta_large':
        plm_name = 'microsoft/deberta-v3-large'
        t_model = 'bert'
    elif arg.plm == 'roberta_large':
        plm_name = "roberta-large"
        t_model = 'roberta'
    elif arg.plm == 'roberta':
        plm_name = "roberta-base"
        t_model = 'roberta'

    if arg.data == 'fb13':
        in_paths = {
            'dataset': arg.data,
            'train': './data/FB13/train.tsv',
            'valid': './data/FB13/dev.tsv',
            'test': './data/FB13/test.tsv',
            'text': ['./data/FB13/entity2text.txt', './data/FB13/relation2text.txt']
        }
    elif arg.data == 'umls':
        in_paths = {
            'dataset': arg.data,
            'train': './data/umls/train.tsv',
            'valid': './data/umls/dev.tsv',
            'test': './data/umls/test.tsv',
            'text': ['./data/umls/entity2textlong.txt', './data/umls/relation2text.txt']
        }
    elif arg.data == 'fb15k-237':
        in_paths = {
            'dataset': arg.data,
            'train': './data/fb15k-237/train.tsv',
            'valid': './data/fb15k-237/dev.tsv',
            'test': './data/fb15k-237/test.tsv',
            'text': ['./data/fb15k-237/FB15k_mid2description.txt',
					 # './data/fb15k-237/entity2textlong.txt',
					 './data/fb15k-237/relation2text.txt']
        }
    elif arg.data == 'wn18rr':
        in_paths = {
            'dataset': arg.data,
            'train': './data/WN18RR/train.tsv',
            'valid': './data/WN18RR/dev.tsv',
            'test': './data/WN18RR/test.tsv',
            'text': ['./data/WN18RR/my_entity2text.txt',
                     './data/WN18RR/relation2text.txt']
        }

    lm_config = AutoConfig.from_pretrained(plm_name, cache_dir='./cached_model')
    lm_tokenizer = AutoTokenizer.from_pretrained(plm_name, do_basic_tokenize=False, cache_dir='./cached_model')
    lm_model = AutoModel.from_pretrained(plm_name, config=lm_config, cache_dir='./cached_model')


    pre_data = preprocess_data(in_paths, lm_tokenizer, batch_size=arg.batch_size,
                             max_desc_length=arg.max_desc_length,
                             model=t_model)



    model = ReAuKGC(lm_model, n_ent=len(pre_data.ent2id), n_rel=len(pre_data.rel2id),
                 )
    model = model.cuda()




    no_decay = ["bias", "LayerNorm.weight"]
    param_group = [
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (not any(nd in n for nd in no_decay))],
         'weight_decay': arg.weight_decay},
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (any(nd in n for nd in no_decay))],
         'weight_decay': 0.0},
    ]

    if not arg.prefix_tuning:
        param_group += [
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (not any(nd in n for nd in no_decay))],  # name中不包含bias和LayerNorm.weight
             'weight_decay': arg.weight_decay},
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0},
        ]

    optimizer = AdamW(param_group)  # transformer AdamW

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= 10 * pre_data.step_per_epc, num_training_steps=arg.epoch * pre_data.step_per_epc)

    hyperparams = {
        'batch_size': arg.batch_size,
        'epoch': arg.epoch,
        'identifier': identifier,
        'load_path': arg.load_path,
        'evaluate_every': 1,
        'update_every': 1,
        'load_epoch': arg.load_epoch,
        'load_metric': arg.load_metric,
        'prefix_tuning': arg.prefix_tuning,
        'plm': arg.plm,
        'description': arg.description,
        'max_desc_length': arg.max_desc_length,
        'wandb': arg.wandb,
        'top_m': arg.top_m
    }

    trainer = Trainer(pre_data, model, lm_tokenizer, optimizer, scheduler, device, hyperparams)

    if arg.link_prediction:
        trainer.link_prediction(split='test')
    else:
        trainer.run()

