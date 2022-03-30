import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn

def adentropy(out_t1, lamda, reduce=True):
    out_t1 = F.softmax(out_t1, dim=1)
    loss_adent = lamda * torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1)
    if reduce:
        loss_adent = torch.mean(loss_adent, dim=0)

    return loss_adent

def info_nce_loss(features, batch_size, device, n_views=2, temperature=0.07):
    
    assert n_views == 2, "Only two view training is supported."

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


def soft_entropy(out, pred, lamda=1, reduce=True):
    out = F.softmax(out, dim=1)
    loss_ent = -lamda * torch.sum(pred * (torch.log(out + 1e-5)), 1)
    if reduce:
        loss_ent = torch.mean(loss_ent, dim=0)

    return loss_ent

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = log_probs.new_zeros(log_probs.shape).scatter_(1, targets.unsqueeze(1), 1)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss

def cross_entropy_label_smooth(pred, target, eps):
    nclass = pred.shape[1]
    loss_func = CrossEntropyLabelSmooth(
        num_classes=nclass,
        epsilon=eps,
        reduction=True
    ).cuda()

    return loss_func(pred, target)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='sum')

def Margin_loss(intensor,label,margin=1,keepdim=False,weight=None):

    onehot_label = F.one_hot(label,intensor.shape[1])
    weight = torch.ones_like(intensor)

    assert weight.shape == intensor.shape

    if not keepdim:
        loss = torch.sum(weight * \
        torch.clamp(margin - (torch.sum(onehot_label * intensor,1).unsqueeze(1) * torch.ones_like(intensor) - intensor)\
        -onehot_label,min=0.0))/(intensor.shape[0]*intensor.shape[1])
    else:
        loss = torch.sum(weight * \
        torch.clamp(margin - (torch.sum(onehot_label * intensor,1).unsqueeze(1) * torch.ones_like(intensor) - intensor)\
        -onehot_label,min=0.0),1,keepdim=True)/(intensor.shape[0]*intensor.shape[1])

    return loss

# if __name__ == '__main__':
#     a = torch.rand(5,10).requires_grad_()
#     b = torch.randn(5,10).requires_grad_()
#     c = torch.LongTensor([0,0,0,0,0])

#     loss1 = F.multi_margin_loss(a,c)
#     loss2 = F.multi_margin_loss(b,c)
#     loss3 = margin_loss(a,c)
#     loss4 = margin_loss(b,c)

#     print(loss1,loss2)
#     print(loss3,loss4)

#     print(torch.autograd.grad(loss1,a,retain_graph=True)[0]==torch.autograd.grad(loss3,a,retain_graph=True)[0])
#     print(torch.autograd.grad(loss2,b,retain_graph=True)[0]==torch.autograd.grad(loss4,b,retain_graph=True)[0])

