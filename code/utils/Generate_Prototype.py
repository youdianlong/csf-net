"""
Prototype Generatation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def getPrototype(fts, mask, region=False):
    """
    Average the features to obtain the prototype

    Args:
        fts: input features, expect shape: B x Channel x X x Y x Z
        mask: binary mask, expect shape: B x class x X x Y x Z
        region: focus region, expect shape: B x X x Y x Z
    """
    # adjust the features H, W shape
    fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')  # 3D uses tri, 2D uses bilinear
    # print(fts.shape)
    # print(mask.shape)
    # masked average pooling
    mask_new = mask.unsqueeze(1)  # bs x 1 x Z x H x W
    # get the masked features
    masked_features = torch.mul(fts, mask_new)  # here use a broadcast mechanism: https://blog.csdn.net/da_kao_la/article/details/87484403
    masked_fts = torch.sum(masked_features*region, dim=(2, 3)) / ((mask_new*region).sum(dim=(2, 3)) + 1e-5)  # bs x C
    # print(sum1.shape)
    # print('masked fts:', masked_fts.shape)
    return masked_fts

def getFeatures(fts, mask, region=False):
    """
    Extract foreground and background features via masked average pooling

    Args:
        fts: input features, expect shape: C x X' x Y' x Z'
        mask: binary mask, expect shape: X x Y x Z
    """
    fts = torch.unsqueeze(fts, 0)
    if torch.is_tensor(region):
        mask = torch.unsqueeze(mask * region, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2,3))
    else:
        mask = torch.unsqueeze(mask, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
    return masked_fts

def calDist(fts, prototype, scaler=1.):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x X x Y x Z
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
    return dist

def calDist_every(prototype1, prototype2, scaler=1.):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x X x Y x Z
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist=0
    for i in range(4):
        dist += 1-F.cosine_similarity(prototype1[i], prototype2[i]) * scaler
    return dist