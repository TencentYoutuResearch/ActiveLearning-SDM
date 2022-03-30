import torch
import numpy as np
from .strategy import Strategy
import torch.nn.functional as F

# use permutation to implement random
# seed is stable
class RandomSampling(Strategy):
    def __init__(self, source, target_train, target_test, idxs_lb, net, cfg):
        super(RandomSampling, self).__init__(source, target_train, target_test, idxs_lb, net, cfg)
        global device;device = torch.device("cuda:" + self.cfg.DEVICE if self.cfg.USE_CUDA else "cpu")

    def query(self,num_active, candidate_dataset):
        return np.random.permutation(len(candidate_dataset))[:num_active]


