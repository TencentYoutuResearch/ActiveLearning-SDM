import numpy as np
from torch import nn
import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy

from util.losses import adentropy, info_nce_loss, NT_XentLoss, SCANLoss, entropy

# Parent class of sample strategy
class Strategy:
    def __init__(self, source, target_train, target_test, idxs_lb, net, cfg):
        self.source = source
        self.target_train = target_train
        self.target_test = target_test
        self.idxs_lb = idxs_lb
        self.net = net
        self.cfg = cfg
        self.n_pool = len(target_train)
        global device;device = torch.device("cuda:" + self.cfg.DEVICE if self.cfg.USE_CUDA else "cpu")

        # self.scanloss = SCANLoss(5)

    # Implement in subclass
    def query(self,n):
        pass
    # update labeled pool
    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, loader_tr, optimizer, epoch):
        self.clf.train()
        for batch_idx, (data, target, path, _) in enumerate(loader_tr):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            feature, output = self.clf(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.cfg.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader_tr.dataset),
                            100. * batch_idx / len(loader_tr), loss.item()))
        print('-----------------------------------------------------------')

    
    # DA trainer
    def train_source_da(self,epoch):
        self.clf =  self.net.to(device)
        optimizer = optim.Adadelta(self.clf.parameters_list(self.cfg.LEARN_RATE), lr=self.cfg.LEARN_RATE)
        self._train(self.source, optimizer, epoch)

    
    # labeled target and unlabeled target accomplished in respective strategy classes
    def train_target(self):
        pass

    # add MME method
    def train_MME(self, epoch):
        self.clf =  self.net.to(device)
        optimizer = optim.Adadelta(self.clf.parameters_list(self.cfg.LEARN_RATE), lr=self.cfg.LEARN_RATE)
        self.clf.train()
        for batch_idx, (data, target, path, _) in enumerate(self.source):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            feature, output = self.clf(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.cfg.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.source.dataset),
                            100. * batch_idx / len(self.source), loss.item()))
        self.clf.eval()
        optimizer_mme = optim.Adadelta(self.clf.parameters_list(0.01),lr=0.01)
        for batch_idx, (data, target, path, _) in enumerate(self.target_train):
            data, target = data.to(device), target.to(device)
            # optimizer.zero_grad()
            feature, output = self.clf(data, reverse=True, eta=self.cfg.REVERSE_WEIGHT)
            ent_loss = adentropy(output, lamda=self.cfg.MME_LAMBDA)
            ent_loss.backward()
            if batch_idx % self.cfg.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMME Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.target_train.dataset),
                            100. * batch_idx / len(self.target_train), ent_loss.item()))
        optimizer_mme.step()
        optimizer_mme.zero_grad()
        print('-----------------------------------------------------------')
        optimizer.step()


    # Label the selected sample and add it to the source domain data 
    def active(self, inds, candidate_dataset, aim_dataset):
        active_samples = list()
        for i in inds:
            _, target, path, _ = candidate_dataset[i]
            active_samples.append([path,target])
        # self.idxs_lb[inds] = True

        aim_dataset.add_item(active_samples)
        candidate_dataset.remove_item(inds)


    
