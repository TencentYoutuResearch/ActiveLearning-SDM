import numpy as np
import sys
from torch.autograd import Function
from sklearn.metrics import pairwise_distances

# init_label_pool
# Not used
def init_label_pool(total_num,init_num):
    idxs_lb = np.zeros(total_num, dtype=bool)
    idxs_tmp = np.arange(total_num)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:init_num]] = True

    return idxs_lb

# different dataset with different prefix
# officehome:Art, Clip, Product, Real_world
# domainnet:Clipart, Infograph, Painting, Quickdraw, Real, Sketch
# office31:Amazon, DSLR, Webcam
def completion_name(prefix, whichset):
    if whichset.lower() == 'officehome':
        if prefix.lower() == 'a':
            return 'Art'
        elif prefix.lower() == 'c':
            return 'Clipart'
        elif prefix.lower() == 'p':
            return 'Product'
        elif prefix.lower() == 'r':
            return 'Real_World'
        else:
            raise ValueError
    elif whichset.lower() == 'domainnet':
        if prefix.lower() == 'c':
            return 'clipart'
        elif prefix.lower() == 'i':
            return 'infograph'
        elif prefix.lower() == 'p':
            return 'painting'
        elif prefix.lower() == 'q':
            return 'quickdraw'
        elif prefix.lower() == 'r':
            return 'real'
        elif prefix.lower() == 's':
            return 'sketch'
        else:
            raise ValueError
    elif whichset.lower() == 'office31':
        if prefix.lower() == 'a':
            return 'amazon'
        elif prefix.lower() == 'w':
            return 'webcam'
        elif prefix.lower() == 'd':
            return 'dslr'
        else:
            raise ValueError

# AADA grad reverse method in to tackle DA problem
class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, lambd = 1.0):
        ctx.constant = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.constant
        return (grad_output * -lambd), None

# Not used
def grad_reverse(x, lambd = 1.0):
    return GradReverse.apply(x, lambd)

