import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.5):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def weight_norm(model):
    w_data_a = model.fc3.fc1.weight.data
    w_data_b = model.fc3.fc2.weight.data

    L_norm_a = torch.norm(w_data_a, p=2, dim =1)
    L_norm_b = torch.norm(w_data_b, p=2, dim =1)
    #print ('norm_a:', L_norm_a, ' norm_b: ', L_norm_b)
    print ('norm_a mean: ', L_norm_a.mean(0), ' norm_b norm: ', L_norm_b.mean(0))
    return L_norm_a.mean(0)/L_norm_b.mean(0)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam