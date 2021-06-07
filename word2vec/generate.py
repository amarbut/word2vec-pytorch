# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:59:57 2021

@author: Anna
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import argparse
import os
import json

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)


def model_fn(model_dir):
    with open(os.path.join(model_dir, 'model_info.pth'), 'rb') as f:
        model_info = torch.load(f)
    model = SkipGramModel(emb_size = model_info['emb_size'], emb_dimension = model_info['emb_dimension'])
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    embeddings = model.u_embeddings.weight().data.numpy()
    word2id = model_info['word2id']
    return {'embeddings':embeddings, 'word2id':word2id}

def input_fn(request_body, request_content_type = 'application/json'):
    if request_content_type == 'application/json':
        return json.loads(request_body)
    raise Exception(request_content_type, ": input type not supported")
    
    
def predict_fn(input_data, model):
    word2id = model['word2id']
    ids = []
    for w in input_data['words']:
        ids.append(word2id[w])
    embeddings = model['embeddings']
    response = dict()
    for i,w in enumerate(input_data):
        response[w] = embeddings[word2id[i]]
    return response

def output_fn(prediction, accept = 'application/json'):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise Exception(accept, ': content type not supported')