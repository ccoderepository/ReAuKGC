import math
import os
# import pdb
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import pickle
import random
import copy
import pdb

num_deg_features = 2


class ReAuKGC(nn.Module):
    def __init__(self, lm_model, n_ent, n_rel):
        super().__init__()

        self.lm_model_given = lm_model
        self.lm_model_target = copy.deepcopy(lm_model)

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.hidden_size = lm_model.config.hidden_size


        self.rel_embeddings = torch.nn.Embedding(n_rel, self.hidden_size)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        # nn.init.xavier_normal_()

        self.classifier = torch.nn.Linear(self.hidden_size, 2)

        self.ent_classifier = torch.nn.Linear(self.hidden_size, n_ent)

        self.sim_classifier = nn.Sequential(nn.Linear(self.hidden_size * 4 + num_deg_features, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size, 1))

    def forward(self, inputs, positions):

        batch_size = len(positions)

        lm_model = self.lm_model_given

        device = lm_model.device

        cls_idx = torch.LongTensor([positions[i]['cls_tokens'][0] for i in range(batch_size)]).to(device)
        cls_pos = torch.LongTensor([positions[i]['cls_tokens'][1] for i in range(batch_size)]).to(device)

        input_ids = inputs.pop('input_ids')
        input_embeds = self.lm_model_given.embeddings.word_embeddings(input_ids).squeeze(1)
        for i in range(batch_size):
            input_embeds[i, cls_pos[i], :] = self.rel_embeddings(cls_idx[i])
        inputs['inputs_embeds'] = input_embeds
        logits = lm_model(**inputs)
        cls_emb_list = []
        for i in range(batch_size):
            cls_emb_list.append(logits[0][i, cls_pos[i], :].unsqueeze(0))
        csl_embs = torch.cat(cls_emb_list, dim=0)
        # torch.nn.functional.normalize(csl_embs, dim=-1)
        return csl_embs


    def encode_target(self, inputs, positions):

        input_ids = inputs.pop('input_ids')
        input_embeds = self.lm_model_given.embeddings.word_embeddings(input_ids).squeeze(1)
        inputs['inputs_embeds'] = input_embeds

        logits = self.lm_model_target(**inputs)
        target_embs = torch.mean(logits[0], dim=1)
        # target_embs = logits[0][:, 1, :]
        # torch.nn.functional.normalize(target_embs, dim=-1)
        return target_embs

    def match(self, target_preds, target_encoded, triple_degrees, test=False, ent_list_degrees=None):
        device = self.lm_model_given.device
        sim = torch.zeros(target_preds.shape[0], target_encoded.shape[0]).to(self.lm_model_given.device)

        if not test:
            assert (ent_list_degrees == None)

        for it, target_pred in enumerate(target_preds):

            triple_degree = triple_degrees[it]
            h_deg, r_deg, t_deg = torch.tensor(triple_degree).float().to(device)
            h_deg, r_deg, t_deg = h_deg.unsqueeze(0), r_deg.unsqueeze(0), t_deg.unsqueeze(0)

            if not test:
                ent_list_degrees = [deg[-1] for deg in triple_degrees]

            t_deg = torch.tensor(ent_list_degrees).float().to(device).unsqueeze(1)
            h_deg = h_deg.expand(target_encoded.shape[0], 1)

            h_logdeg, r_logdeg, t_logdeg = (h_deg + 1).log(), (r_deg + 1).log(), (t_deg + 1).log()

            deg_feature = torch.cat([t_logdeg, h_logdeg], dim=-1)

            target_pred = target_pred.expand(target_encoded.shape[0], target_pred.shape[0])

            sim[it] = self.sim_classifier(torch.cat(
                [target_pred, target_encoded, target_pred - target_encoded, target_pred * target_encoded, deg_feature],
                dim=-1)).T
        return sim

