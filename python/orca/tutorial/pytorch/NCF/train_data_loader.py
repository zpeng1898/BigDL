#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Most of the pytorch code is adapted from guoyang9's NCF implementation for
# ml-1m dataset.
# guoyang9's source code: https://github.com/guoyang9/NCF
# MovieLens 1M Dataset: https://grouplens.org/datasets/movielens/1m/
#

import os
import time
import argparse
import numpy as np
import pandas as pd 
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F 
from model import NCF

#Step 0: Parameters And Configuration

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", 
    type=str, 
    default="ml-1m", 
    help="dataset name")
parser.add_argument("--model", 
    type=str, 
    default="NeuMF-end", 
    help="model name")
parser.add_argument("--num_ng", 
    type=int, 
    default=4, 
    help="sample negative items for training")
parser.add_argument("--backend", 
    type=str, 
    default="ray", 
    help="backend used in estimator, ray or spark are supported")
args = parser.parse_args()

#Step 1: Init Orca Context

from bigdl.orca import init_orca_context, stop_orca_context
init_orca_context(cores=1, memory="8g") # 1 cpu core

#Step 2: Define Train Dataset

from sklearn.model_selection import train_test_split

def load_all():
    """ We load all the files here to save time in each epoch. """
    data_X = pd.read_csv(
        args.dataset+"/ratings.dat", 
        sep="::", header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    
    user_num = data_X['user'].max() + 1
    item_num = data_X['item'].max() + 1

    data_X = data_X.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in data_X:
        train_mat[x[0], x[1]] = 1.0
        
    train_data, test_data = train_test_split(data_X, test_size=0.1, random_state=100)
    
    return train_data, test_data, user_num, item_num, train_mat

class NCFData(data.Dataset):
    def __init__(self, features, 
                num_item, train_mat=None, num_ng=4, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)
    
    def __getitem__(self, idx):
        features =  self.features_fill if self.is_training else self.features_ps
        labels =  self.labels_fill if self.is_training else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = float(labels[idx])
        return [user, item] ,label

# prepare the train and test datasets
train_data, test_data, args.user_num, args.item_num, train_mat = load_all()

#Step 3: Define the Model

# create the model
def model_creator(config):
    model = NCF(args.user_num, args.item_num, factor_num=32, num_layers=3, dropout=0.0, model=args.model) # a torch.nn.Module
    model.train()
    return model

#create the optimizer
def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=0.001)

#define the loss function
loss_function = nn.BCEWithLogitsLoss()

#Step 4: Fit with Orca Estimator

from bigdl.orca.learn.pytorch import Estimator 
from bigdl.orca.learn.metrics import Accuracy

# create the estimator
est = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator,loss=loss_function, metrics=[Accuracy()],backend=args.backend)# backend="ray" or "spark"

# construct the train and test dataloader
def train_loader_func(config, batch_size):
    train_dataset = NCFData(train_data, args.item_num, train_mat, num_ng=4, is_training=True)
    train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    train_loader.dataset.ng_sample()# sample negative items for training datasets
    return train_loader

def test_loader_func(config, batch_size):
    test_dataset = NCFData(test_data, args.item_num, train_mat, num_ng=0, is_training=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    return test_loader

# fit the estimator
est.fit(data=train_loader_func, epochs=5)

#Step 5: Save and Load the Model

# save the model
est.save("NCF_model")

# load the model
est.load("NCF_model")

# evaluate the model
result = est.evaluate(data=test_loader_func)
for r in result:
    print(r, ":", result[r])

# stop orca context when program finishes
stop_orca_context()