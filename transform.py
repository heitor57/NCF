import torch

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import numpy as np
import model
import config
import evaluate
import data_utils
model = torch.load('{}{}_{}.pth'.format(config.model_path, config.model,config.dataset))

with open('user_feature.npy', 'wb') as f:
	# np.save(f,np.array(model.state_dict()['embed_user_GMF.weight']))
	# print(np.array(model.state_dict()['embed_user_GMF.weight']).shape)
	# print(np.array([[0]*256]).shape)
	# a = np.vstack((np.array([[0]*256]),np.array(model.state_dict()['embed_user_GMF.weight'])))
	a = np.array(model.state_dict()['embed_user_GMF.weight'])
	a=a.astype('float32')
	np.save(f,a)
with open('item_feature.npy', 'wb') as f:
	a = np.array(model.state_dict()['embed_item_GMF.weight'])
	a=a.astype('float32')
	np.save(f,a)

# np.array(model.state_dict()['embed_item_GMF.weight'])
# print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
