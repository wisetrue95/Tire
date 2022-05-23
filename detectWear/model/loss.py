import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def cross_entropy(output, target):
    loss=CrossEntropyLoss()
    entropy_loss=loss(output, target)
    return entropy_loss

# def cross_entropy(output, target, reduction='mean'):
#     return F.cross_entropy(output, target, reduction=reduction)
