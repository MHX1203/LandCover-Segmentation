import torch
import torch.nn as nn
from torch.nn import functional as F

# class CrossEntropyWithLabelSmooth(nn.Module):

#     def __init__(self,  smooth=0.1,  weight=None,  size_average='mean'):
#         self.smooth = smooth
        
#         if weight is not None:
#             self.weight = t.tensor(weight).float()
            
#         self.size_average = size_average

#     def forward(self,  mask,  pred):
#         """

#         1 / (k - 1)
#         """
#         nn.CrossEntropyLoss()
#         pass
    


class FocalLoss(nn.Module):
    def __init__(self,  gamma=0,  alpha=None,  size_average=True):
        super(FocalLoss,  self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): 
            self.alpha = torch.Tensor(alpha)

        self.size_average = size_average

    def forward(self,  input,  target):
        """
        
         alpha * (1 - p_t) ^ gamma * log(p_t)

         p_t = p_t if y == 1
         p_t = 1 - p_t if y != 1
         
        """

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N, C, H, W => N, C, H*W
            input = input.transpose(1, 2)    # N, C, H*W => N, H*W, C
            input = input.contiguous().view(-1, input.size(2))   # N, H*W, C => N*H*W, C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input,  1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()