import numpy as np
import torch
from .RandomSample import RandomSampling
from .SDM import SDM

# load different sample strategy
def load_strategy(strategy, source, target_train, target_test, idxs_lb, net, cfg):
    if strategy.lower() == 'random':
        # random
        return RandomSampling(source, target_train, target_test, idxs_lb, net, cfg)
    elif strategy.lower() == 'sdm':
        # sdm
        return SDM(source, target_train, target_test, idxs_lb, net, cfg)
    else:
        raise ValueError
