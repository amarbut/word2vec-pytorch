# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:29:59 2021

@author: Anna
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import init
import argparse
import os
import numpy as np
import pickle
import json

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, train_dir, input_file_name):

        self.negative_probs = []
        self.subsample_probs = dict()

        self.word2id = dict()
        self.id2word = dict()
        self.token_count = 0
        self.sentence_count = 0
        self.word_frequency = dict()

        self.input_file_name = os.path.join(train_dir, input_file_name)
        self.read_words()
        self.init_negative_probs()
        self.init_subsample_probs()

#get term counts over entire dataset    
    def read_words(self):
        
        word_frequency = dict()
        for line in open(self.input_file_name, encoding = 'utf8'):
            line = line.split()
            if len(line) > 1:
                self.sentence_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print(str(int(self.token_count / 1000000)), "M tokens ingested.")

        wid = 0
        for w, c in word_frequency.items():
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total unique words: ", str(len(self.word2id)))

#calculate subsample probability according to 2nd Word2Vec paper equation
    def init_subsample_probs(self):
        t = 0.00001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.subsample_probs = 1-np.sqrt(t / f)       

#create array of words to be used in negative samples of size NEGATIVE_TABLE_SIZE; sample with adjusted unigram method
    def init_negative_probs(self):
        
        #use adjusted unigram sampling--raise the word frequencies to the 3/4 power to make less frequent words appear more often
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        
        #calculate how many times ea word (rep by its wid) should appear in the neg sample array based on adjusted freq above
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negative_probs += [wid] * int(c)
        self.negative_probs = np.array(self.negative_probs)
        
        #randomize array of words to be sampled from in data set creation
        np.random.shuffle(self.negative_probs)    
        
#function to sample from negative sampling list        
    def get_negatives(self, pos, size, boundary):
        u,v = pos
        num_negs = size*boundary
        #collect list of neg sample words
        response = []
        while len(response) < num_negs:
            response.appextendend([i for i in np.random.choice(self.negative_probs, num_negs) if i != u and (i not in v)])
        #return list of neg sample words
        return response[0:num_negs] 
    
    
# -------------------------------------------------------------------------------------------------------------------------
        
class Word2vecDataset:
    def __init__(self, data, window_size, wordnet_sim_dict = None, wn_negative_sample = False, wn_positive_sample = False):
        self.data = data
        self.wordnet = wordnet_sim_dict
        self.window_size = window_size
        self.input_file = data.input_file_name
        self.wn_negative_sample = wn_negative_sample
        self.wn_positive_sample = wn_positive_sample

    def __len__(self):
        return self.data.sentences_count
    
    #function used to retrieve all 
    def __getitem__(self, idx):
        while True:
            with open(self.input_file, encoding = 'utf-8') as f:
                line = f.readline()
                if not line:
                    f.seek(0, 0)
                    line = f.readline()
    
                if len(line) > 1:
                    words = line.split()
    
                    if len(words) > 1:
                        #collect word ids for sentence, ignoring words w/ subsample probability
                        word_ids = [self.data.word2id[w] for w in words if
                                    w in self.data.word2id and np.random.rand() > self.data.subsample_probs[self.data.word2id[w]]]
                        
                        #context window ranges from 1 to window size
                        boundary = np.random.randint(1, (self.window_size-1)/2)
                        
                        #collect list of all target/positive context pairs
                        pos_pairs = [(u,word_ids[max(i - boundary, 0):i + boundary]) for i, u in enumerate(word_ids)]
                        
                        #do vanilla word2vec neg sampling OR throw exception if mismatched parameters
                        if not self.wordnet:
                            if self.wn_negative_sample or self.wn_positive_sample:
                                raise Exception("Need Wordnet similarity dict to perform Wordnet function.")
                            else:
                                #vanilla negative sampling method from 2nd word2vec paper
                                negs = [self.data.get_negatives(pos,5, boundary*2) for pos in pos_pairs]
                        
                        #include wordnet data in positive and/or negative samples
                        else:
                            
                            if self.wn_positive_sample:
                            #replace target word with wordnet similar words and add to positive examples
                                wn_pairs = []
                                for pos in pos_pairs:
                                    u,v = pos
                                    w = self.data.id2word[u]
                                    syns = self.wordnet[w]
                                    
                                    #subsample wordnet similar words before adding
                                    wn_pairs.extend([(self.data.word2id[w],v) for w in syns
                                                         if np.random.rand() > self.data.subsample_probs[self.data.word2id[w]]])
                                pos_pairs.extend(wn_pairs)
                
                            if self.wn_negative_sample:
                            #use one wordnet similar word per positive context as negative context
                            #to provide positive wordnet similarity samples
                                negs = []
                                for pos in pos_pairs:
                                    u,v = pos
                                    w = self.data.id2word[u]
                                    
                                    #gets similar words and convert to ids
                                    syns = self.wordnet[w]
                                    sids = [self.data.word2id[w] for w in syns]
                                    
                                    #select similar words based on negative sampling probs
                                    syn_probs = np.array([self.data.negative_probs[i] for i in sids])
                                    syn_norm = syn_probs/sum(syn_probs)
                                    
                                    neg = [i for i in np.random.choice(sids, boundary*2, p = syn_norm)]
                                    neg.extend(self.data.get_negatives(pos, 4, boundary*2))
                                    negs.append(neg)
                            else:
                            #vanilla negative sampling method from 2nd word2vec paper
                                negs = [self.data.get_negatives(pos,5, boundary*2) for pos in pos_pairs]
                        return([(pair[0],pair[1],negs[i]) for i,pair in enumerate(pos_pairs)])
                        

    @staticmethod
    #combine all target, context, and negative samples into tensors for each batch
    def collate_fn(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg = [neg for batch in batches for _, _, neg in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg)

#---------------------------------------------------------------------------------------------------------------------------

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_dimension, wn_negative_sample, wordnet_sim_dict, id2word):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.wn_negative_sample = wn_negative_sample
        self.wordnet_sim_dict = wordnet_sim_dict
        self.id2word = id2word
        
        #initialize target and context embeddings with vocab size and embedding dimension
        self.u_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        
        initrange = 1.0 / self.emb_dimension
        #initialize target embeddings with random nums between -/+ initrange
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        #initialize context embeddings with zeros--first values will be determined by loss update after first batch
        init.constant_(self.v_embeddings.weight.data, 0)
        
    def forward(self, u, v, neg):
        #get input embeddings for target, context, and negative samples
        emb_u = self.u_embeddings(u)
        emb_v = self.v_embeddings(v)
        emb_neg = self.v_embeddings(neg)
        
        #calculate dot product for target and all context words
        pos = torch.bmm(emb_v,emb_u.unsqueeze(2)).squeeze()
        #pos = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        #use sigmoid function to convert to "probability", then subtract from 1 to get loss
        #(we expect 1 for positive context words)
        pos = 1-torch.sigmoid(pos)
        
        #calculate dot product for target and all negative sample words
        neg = torch.bmm(emb_neg,emb_u.unsqueeze(2)).squeeze()
        #use sigmoid function to convert to "probability", then make negative to get loss
        #(we expect 0 for positive context words)
        neg = 1-torch.sigmoid(neg)
        
        #add in wordnet similarity contrastive loss
        if self.wn_negative_sample:
            if not self.wordnet_sim_dict:
                raise Exception("Need Wordnet similarity dict to perform Wordnet function.")
            
            all_sim = []
            all_not_sim = []
            
            #sort context and neg sample words by wordnet similarity to target
            for i in u:
                w = self.id2word[i]
                sim = []
                not_sim = []
                for j in v:
                    vw = self.id2word[j]
                    if vw in self.wordnet_sim_dict[w]:
                        sim.append(j)
                    else:
                        not_sim.append(j)
                for k in neg:
                    negw = self.id2word[k]
                    if negw in self.wordnet_sim_dict[w]:
                        sim.append(k)
                    else:
                        not_sim.append(k)
                all_sim.append(sim)
                all_not_sim.append(not_sim)
            
            #get embeddings for wordnet similarity groups
            emb_sim = self.v_embeddings(torch.LongTensor(all_sim))
            emb_not_sim = self.v_embeddings(torch.LongTensor(all_not_sim))
            
            
            #compute euclidean distance between target word and similar wn words
            #consider using inner product instead?
            wn_loss = 0.0
            wnpos = []
            for i in emb_u:
                wnpos.append((((emb_u[i]-emb_sim[i])**2).sum(1))**(1/2))
                
            
            
            wnneg = []
            for i in emb_u:
                wnpos.append((((emb_u[i]-emb_not_sim[i])**2).sum(1))**(1/2))
    
        #return average loss for each target word (average loss from context and negative samples)
        return torch.mean(torch.cat(pos,neg,1))
        
                    



