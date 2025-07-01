import torch
import torch.nn as nn
import torch.nn.functional as F

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    # print(tensor.max())
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor.long(), 1)
    return one_hot


def weight_self_pro_softmax_mse_loss(input_logits, target_logits,entropy):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    target_logits = target_logits .view(target_logits .size(0), target_logits .size(1), -1)
    target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=2)
    target_softmax = F.softmax(target_logits, dim=2)
    mse_loss = (input_softmax.detach()-target_softmax)**2
    #entropy =  1-entropy.unsqueeze(-1).detach()
    #mse_loss =entropy*mse_loss
    return mse_loss
def weight_cross_pro_softmax_mse_loss(weight,input_logits, target_logits): ##target_logits==classfier input_logit = cross_prototype
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = F.softmax(weight,dim=2)
    target_logits = target_logits .view(target_logits .size(0), target_logits .size(1), -1)
    target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
    assert input_logits.size() == target_logits.size()
    #input_softmax = F.softmax(input_logits, dim=2)
    target_softmax = F.softmax(target_logits, dim=2)
    mse_loss = (input_logits.detach()-target_softmax)**2
    mse_loss = weight.detach()*mse_loss
    return mse_loss
# def double_weight_cross_pro_softmax_mse_loss(weight,input_logits, target_logits,entropy): ##target_logits==classfier input_logit = cross_prototype
#     """Takes softmax on both sides and returns MSE loss
#     Note:
#     - Returns the sum over all examples. Divide by the batch size afterwards
#       if you want the mean.
#     - Sends gradients to inputs but not the targets.
#     """
#     weight = F.softmax(weight,dim=2)
#     weight = torch.max(weight,dim=2,keepdim=True)[0]
#     target_logits = target_logits.view(target_logits .size(0), target_logits .size(1), -1)
#     target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
#     assert input_logits.size() == target_logits.size()
#     #input_softmax = F.softmax(input_logits, dim=2)
#     target_softmax = F.softmax(target_logits, dim=2)
#     mse_loss = (input_logits.detach()-target_softmax)**2
#     mse_loss = weight.detach()*mse_loss
#     entropy =  1 - entropy.unsqueeze(-1).detach()
#     return  entropy * mse_loss

def double_weight_cross_pro_softmax_mse_loss(weight,input_logits, target_logits,entropy): ##target_logits==classfier input_logit = cross_prototype
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = F.softmax(weight,dim=2)
    weight = torch.max(weight,dim=2,keepdim=True)[0]
    target_logits = target_logits.view(target_logits .size(0), target_logits .size(1), -1)
    target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
    assert input_logits.size() == target_logits.size()
    #input_softmax = F.softmax(input_logits, dim=2)
    target_softmax = F.softmax(target_logits, dim=2)
    mse_loss = (input_logits.detach()-target_softmax)**2
    mse_loss = weight.detach()*mse_loss
    entropy =  1 - entropy.unsqueeze(-1).detach()
    return  entropy * mse_loss
