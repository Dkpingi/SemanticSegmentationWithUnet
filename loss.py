import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def to_one_hot(tensor,nClasses):
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w).cuda().scatter_(1,tensor.view(n,1,h,w),1)
    return one_hot

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target):
    	# inputs => N x Classes x H x W
    	# target_oneHot => N x Classes x H x W
        N = inputs.size()[0]
        target_oneHot = to_one_hot(target,3)
    	# predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)
    	
    	# Numerator Product
        inter = inputs * target_oneHot
    	## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

    	#Denominator 
        union = inputs + target_oneHot - (inputs*target_oneHot)
    	## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)
        
        loss = inter/union

    	## Return average loss over classes and batch
        return 1 - loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.reshape(-1,1)

        logpt = F.log_softmax(input, dim = 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()