import os
import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.mixture import GaussianMixture

import config.load_config as cfg
import data.dataset as loader
import data.transform as transform
from util.utils import init_label_pool
from model.resnet import ResNet50Fc
from sample_strategy.strategy_loader import load_strategy

def main():
    # set random seed
    global device;device = torch.device("cuda:" + cfg.DEVICE if cfg.USE_CUDA else "cpu")
    kwargs = {'num_workers': cfg.NUM_WORK, 'pin_memory': True} if cfg.USE_CUDA else {}
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # load train_source_data train_target_data and test_target_data
    train_source_data = loader.get_data(cfg.DATA_NAME,os.path.join(cfg.DATA_PATH,cfg.DATA_SOURCE),
    transform.train_transform,tr_or_te='train',n_views=cfg.N_VIEWS)
    train_target_data = loader.get_data(cfg.DATA_NAME,os.path.join(cfg.DATA_PATH,cfg.DATA_TARGET),
    transform.test_transform,tr_or_te='train',n_views=cfg.N_VIEWS)
    test_target_data = loader.get_data(cfg.DATA_NAME,os.path.join(cfg.DATA_PATH,cfg.DATA_TARGET),
    transform.test_transform,tr_or_te='test',n_views=cfg.N_VIEWS)

    source_train_loader = DataLoader(train_source_data, batch_size=cfg.BATCH_SIZE, 
    shuffle=True, drop_last=True,**kwargs)
    target_train_loader = DataLoader(train_target_data, batch_size=cfg.BATCH_SIZE,**kwargs)
    target_test_loader = DataLoader(test_target_data, batch_size=cfg.BATCH_SIZE,**kwargs)

    # init label pool
    n_pool = len(test_target_data)
    idxs_lb = init_label_pool(n_pool,cfg.NUM_INIT_LB)
    num_active = math.ceil(n_pool * cfg.QUERY_RATIO)

    # load model
    net = ResNet50Fc(class_num = cfg.DATA_CLASS)
    net2 = ResNet50Fc(class_num = cfg.DATA_CLASS)
    # print(net.fc.weight.shape)

    # select strategy
    strategy = load_strategy(cfg.SAMPLE_STRATEGY, 
    source_train_loader, target_train_loader, target_test_loader, idxs_lb, net, cfg)

    print('-----------------------------------------------------------')
    print('Start Sample Strategy %s with data %s --> %s'%(type(strategy).__name__,cfg.DATA_SOURCE,cfg.DATA_TARGET))
    print('-----------------------------------------------------------')

    for epoch in range(1,cfg.EPOCH+1):
        strategy.train_SDM(epoch)
        if epoch in [10, 12, 14, 16, 18]:
            # query samples with different active learning strategy
            query_indx = strategy.SDM_query(num_active)
            strategy.sdm_active(query_indx, train_target_data, train_source_data)
        strategy.test()



    





if __name__ == "__main__":

    main()
