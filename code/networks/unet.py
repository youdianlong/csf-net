# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function
import sys

sys.path.append("/data2/tcg/csf-net/code/vmamba")
from vmamba.SEEM import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

def masked_average_pooling(feature, mask, reliability_map=None):
    # print(feature.shape[-2:])
    if torch.is_tensor(reliability_map):
        mask = F.interpolate(mask, size=feature.shape[-2:], mode='bilinear', align_corners=True)
        mask = mask*reliability_map[:,None,:,:]
        # print((feature*mask).shape)
        masked_feature = torch.sum(feature * mask, dim=(2, 3))
    else:
        mask = F.interpolate(mask, size=feature.shape[-2:], mode='bilinear', align_corners=True)
        # print((feature*mask).shape)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
    return masked_feature


def batch_prototype(feature, mask, reliability_map=None):  # return B*C*feature_size
    batch_pro = torch.zeros(mask.shape[0], mask.shape[1], feature.shape[1])
    if torch.is_tensor(reliability_map):
        for i in range(mask.shape[1]):
            classmask = mask[:, i, :, :]
            proclass = masked_average_pooling(feature, classmask.unsqueeze(1), reliability_map)
            batch_pro[:, i, :] = proclass
    else:
        for i in range(mask.shape[1]):
            classmask = mask[:, i, :, :]
            proclass = masked_average_pooling(feature, classmask.unsqueeze(1))
            batch_pro[:, i, :] = proclass
    return batch_pro


def entropy_value(p, C):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=2) / torch.tensor(np.log(C))  # .cuda()
    return y1


def agreementmap(similarity_map):
    score_map = torch.argmax(similarity_map, dim=3)
    # score_map =score_map.transpose(1,2)
    ##print(score_map.shape, 'score',score_map[0,0,:])
    gt_onthot = F.one_hot(score_map, 4)
    avg_onehot = torch.sum(gt_onthot, dim=2).float()
    avg_onehot = F.normalize(avg_onehot, 1.0, dim=2)
    ##print(gt_onthot[0,0,:,:],avg_onehot[0,0,:])
    weight = 1 - entropy_value(avg_onehot, similarity_map.shape[3])
    ##print(weight[0,0])
    # score_map = torch.sum(score_map,dim=2)
    return weight


def similarity_calulation(feature, batchpro):  # feature_size = B*C*H*W  batchpro= B*C*dim
    B = feature.size(0)
    feature = feature.view(feature.size(0), feature.size(1), -1)  # [N, C, HW]
    feature = feature.transpose(1, 2)  # [N, HW, C]
    feature = feature.contiguous().view(-1, feature.size(2))
    C = batchpro.size(1)
    batchpro = batchpro.contiguous().view(-1, batchpro.size(2))
    feature = F.normalize(feature, p=2.0, dim=1)
    batchpro = F.normalize(batchpro, p=2.0, dim=1).cuda()
    similarity = torch.mm(feature, batchpro.T)
    similarity = similarity.reshape(-1, B, C)
    similarity = similarity.reshape(B, -1, B, C)
    return similarity

def adjusted_cosine_similarity(feature, batchpro):

    B = feature.size(0)
    feature = feature.view(feature.size(0), feature.size(1), -1)  # [N, C, HW]
    feature = feature.transpose(1, 2)  # [N, HW, C]
    feature = feature.contiguous().view(-1, feature.size(2))
    # feature = feature - torch.mean(feature,dim=-1).view(-1, 1)
    C = batchpro.size(1)
    batchpro = batchpro.contiguous().view(-1, batchpro.size(2))
    # batchpro = batchpro - torch.mean(batchpro, dim=-1).view(-1, 1)
    feature = F.normalize(feature, p=2.0, dim=1)
    batchpro = F.normalize(batchpro, p=2.0, dim=1).cuda()
    similarity = torch.mm(feature, batchpro.T)
    similarity = similarity.reshape(-1, B, C)
    similarity = similarity.reshape(B, -1, B, C)

    return similarity


def selfsimilaritygen(similarity):
    B = similarity.shape[0]
    mapsize = similarity.shape[1]
    C = similarity.shape[3]
    selfsimilarity = torch.zeros(B, mapsize, C)
    for i in range(similarity.shape[2]):
        selfsimilarity[i, :, :] = similarity[i, :, i, :]
    return selfsimilarity.cuda()


def othersimilaritygen(similarity):
    similarity = torch.exp(similarity)
    for i in range(similarity.shape[2]):
        similarity[i, :, i, :] = 0
    similaritysum = torch.sum(similarity, dim=2)
    similaritysum_union = torch.sum(similaritysum, dim=2).unsqueeze(-1)
    # print(similaritysum_union.shape)
    othersimilarity = similaritysum / similaritysum_union
    # print(othersimilarity[1,1,:].sum())
    return othersimilarity


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class InterSampleAttention(torch.nn.Module):
    """
        Implementation for inter-sample self-attention
        input size for the encoder_layers: [batch, h x w x d, dim]
    """

    def __init__(self, input_dim=256, hidden_dim=1024):
        super(InterSampleAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)
        self.mambablock = ResidualBlock(input_dim, input_dim * 2, 1024, 16, 2, dt_rank=None, d_conv=3,
                                        pad_vocab_size_multiple=8, conv_bias=True, bias=False)

        self.pro_in = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.pro_out = nn.Linear(self.hidden_dim, self.input_dim, bias=False)

    def forward(self, feature):
        if self.training:
            b, c, h, w = feature.shape
            feature = feature.permute(0, 2, 3, 1).contiguous()
            feature = feature.view(b, h * w, c)
            feature = feature.permute(1, 0, 2).contiguous()
            # feature = self.encoder_layers(feature)
            # feature = self.pro_in(feature)
            feature = self.mambablock(feature)
            # feature = self.pro_out(feature)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = feature.view(b, h, w, c)
            feature = feature.permute(0, 3, 1, 2).contiguous()
        return feature


class IntraSampleAttention(torch.nn.Module):
    """
    Implementation for intra-sample self-attention
    input size for the encoder_layers: [h x w x d, batch, dim]
    """

    def __init__(self, input_dim=256, hidden_dim=1024):
        super(IntraSampleAttention, self).__init__()
        self.input_dim = input_dim
        # self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)
        self.mambablock = ResidualBlock(input_dim, input_dim * 2, 1024, 16, 2, dt_rank=None, d_conv=3,
                                        pad_vocab_size_multiple=8, conv_bias=True, bias=False)
        self.hidden_dim = hidden_dim
        # self.embedding = nn.Embedding(input_dim, hidden_dim)

        self.pro_in = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.pro_out = nn.Linear(self.hidden_dim, self.input_dim, bias=False)

    def forward(self, feature):
        if self.training:
            b, c, h, w = feature.shape
            feature = feature.permute(0, 2, 3, 1).contiguous()
            feature = feature.view(b, h * w, c)  # （24，1024,128）
            # feature = feature.permute(1, 0, 2).contiguous()
            # feature = self.pro_in(feature)
            feature = self.mambablock(feature)
            # feature = self.pro_out(feature)
            # feature = self.encoder_layers(feature)
            # feature = feature.permute(1, 0, 2).contiguous()
            feature = feature.view(b, h, w, c)
            feature = feature.permute(0, 3, 1, 2).contiguous()
        return feature


class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in // 2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class EncoderAuxiliary(nn.Module):
    def __init__(self, params):
        super(EncoderAuxiliary, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.intra_attention = IntraSampleAttention(input_dim=128, hidden_dim=128 * 4)
        self.inter_attention = InterSampleAttention(input_dim=128, hidden_dim=128 * 4)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # 加了x3之后，涨一个点，指标普遍上涨
        x3 = self.intra_attention(x3) + x3
        x3 = self.inter_attention(x3) + x3


        x4_last = self.down4(x3)
        x4 = self.dropout(x4_last)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x5 = self.up1(x4, x3)

        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.up4(x7, x0)
        output = self.out_conv(x8)
        return output, [x5, x6, x7, x8]


class Decoder_pro(nn.Module):
    def __init__(self, params):
        super(Decoder_pro, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)  # (32,128,128)
        x8 = self.up4(x7, x0)  # (32,256,256)
        # print(x.shape,'feature_shape')
        output = self.out_conv(x8)
        mask = torch.softmax(output, dim=1)
        uncertainty = -torch.sum(mask * torch.log(mask + 1e-16), dim=1)
        norm_uncertainty = torch.stack([uncertain / torch.sum(uncertain) for uncertain in uncertainty], dim=0)
        reliability_map = (1 - norm_uncertainty) / np.prod(np.array(norm_uncertainty.shape[-2:]))
        batch_pro = batch_prototype(x8, mask, reliability_map)
        # similarity_map = adjusted_cosine_similarity(x8, batch_pro)
        similarity_map = similarity_calulation(x8, batch_pro)
        entropy_weight = agreementmap(similarity_map)
        self_simi_map = selfsimilaritygen(similarity_map)  # B*HW*C
        other_simi_map = othersimilaritygen(similarity_map)  # B*HW*C
        return output, self_simi_map, other_simi_map, entropy_weight, [x5, x6, x7, x8]


class SideConv(nn.Module):
    def __init__(self, n_classes=4):
        super(SideConv, self).__init__()

        self.side4 = nn.Conv2d(128, n_classes, 1, padding=0)
        self.side3 = nn.Conv2d(64, n_classes, 1, padding=0)
        self.side2 = nn.Conv2d(32, n_classes, 1, padding=0)
        self.side1 = nn.Conv2d(32, n_classes, 1, padding=0)
        # self.side1 = nn.Conv2d(16, n_classes, 1, padding=0)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

    def forward(self, stage_feat):
        x5_up, x6_up, x7_up, x8_up = stage_feat[0], stage_feat[1], stage_feat[2], stage_feat[3]

        out4 = self.side4(x5_up)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)

        out3 = self.side3(x6_up)
        out3 = self.upsamplex2(out3)
        out3 = self.upsamplex2(out3)

        out2 = self.side2(x7_up)
        out2 = self.upsamplex2(out2)

        out1 = self.side1(x8_up)
        return [out4, out3, out2, out1]


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1


class MCNet2d_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        return output1, output2


class MCNet2d_v2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 2,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        return output1, output2, output3


# class MCNet2d_ours(nn.Module):
#     def __init__(self, in_chns, class_num):
#         super(MCNet2d_ours, self).__init__()
#
#         params1 = {'in_chns': in_chns,
#                    'feature_chns': [16, 32, 64, 128, 256],
#                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                    'class_num': class_num,
#                    'up_type': 1,
#                    'acti_func': 'relu'}
#         params2 = {'in_chns': in_chns,
#                    'feature_chns': [16, 32, 64, 128, 256],
#                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                    'class_num': class_num,
#                    'up_type': 0,
#                    'acti_func': 'relu'}
#         params3 = {'in_chns': in_chns,
#                    'feature_chns': [16, 32, 64, 128, 256],
#                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                    'class_num': class_num,
#                    'up_type': 2,
#                    'acti_func': 'relu'}
#         params4 = {'in_chns': 5,
#                    'feature_chns': [16, 32, 64, 128, 256],
#                    'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                    'class_num': class_num,
#                    'up_type': 1,
#                    'acti_func': 'relu'}
#         self.encoder1 = Encoder(params1)
#         # self.encoder2 = EncoderAuxiliary(params1)
#         self.encoder2 = EncoderAuxiliary(params4)
#         self.decoder1 = Decoder(params1)
#         self.decoder2 = Decoder(params2)
#         self.decoder3 = Decoder(params3)
#         self.DF_Module = DF_Module(4, 4, True).cuda()
#         self.sideconv1 = SideConv()
#
#     #
#     def forward(self, x):
#         feature1 = self.encoder1(x)
#         output1, stage_feat1 = self.decoder1(feature1)
#         output2, stage_feat2 = self.decoder2(feature1)
#         shape_ori = self.DF_Module(output1, output2)
#         shape = F.softmax(shape_ori, dim=1)
#         deep_out1 = self.sideconv1(stage_feat1)
#         # deep_out2 = self.sideconv1(stage_feat2)
#         shape_input = torch.cat((x, shape), dim=1)
#         # feature2 = self.encoder2(x)
#         feature2 = self.encoder2(shape_input)
#         output3, stage_feat3 = self.decoder3(feature2)
#         return output1, output2, output3, shape_ori, deep_out1,[stage_feat1[1],stage_feat2[1],stage_feat3[1]]  # ,deep_out1,deep_out2


class ours_net(nn.Module):
    def __init__(self, in_chns, class_num):
        super(ours_net, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [32, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [32, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                   'feature_chns': [32, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 2,
                   'acti_func': 'relu'}
        # params4 = {'in_chns': 3,
        #            'feature_chns': [32, 32, 64, 128, 256],
        #            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
        #            'class_num': class_num,
        #            'up_type': 1,
        #            'acti_func': 'relu'}
        params4 = {'in_chns': 5,
                   'feature_chns': [32, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        self.encoder1 = Encoder(params1)
        # self.encoder2 = Encoder(params1)
        # self.encoder2 = Encoder(params4)
        # self.encoder2 = EncoderAuxiliary(params1)
        self.encoder2 = EncoderAuxiliary(params4)
        # self.decoder1 = Decoder(params1)
        # self.decoder2 = Decoder(params2)
        # self.decoder3 = Decoder(params3)
        self.decoder1 = Decoder_pro(params1)
        self.decoder2 = Decoder_pro(params2)
        self.decoder3 = Decoder_pro(params3)
        self.DF_Module = DF_Module(class_num, class_num, True).cuda()
        self.sideconv1 = SideConv(n_classes=class_num)

    #
    def forward(self, x):
        feature1 = self.encoder1(x)
        output1, self_simi_map1, other_simi_map1, entropy_weight1, stage_feat1 = self.decoder1(feature1)
        output2, self_simi_map2, other_simi_map2, entropy_weight2, stage_feat2 = self.decoder2(feature1)
        # output1, stage_feat1 = self.decoder1(feature1)
        # output2, stage_feat2 = self.decoder2(feature1)
        # ##########################################################
        shape_ori = self.DF_Module(output1, output2)
        # shape_ori = self.DF_Module(output1, output2)
        # shape = F.softmax(output1, dim=1)
        shape = F.softmax(shape_ori, dim=1)
        deep_out1 = self.sideconv1(stage_feat1)
        # deep_out2 = self.sideconv1(stage_feat2)
        # ##########################################################
        # 消融实验
        shape_input = torch.cat((x, shape), dim=1)

        # feature2 = self.encoder2(x)
        feature2 = self.encoder2(shape_input)
        ###########################################################
        # output3, stage_feat3 = self.decoder3(feature2)
        output3, self_simi_map3, other_simi_map3, entropy_weight3, stage_feat3 = self.decoder3(feature2)

        # return output1, output2, output3#, deep_out1
        # return output1, output2, output3, shape#, deep_out1
        return output1, output2, output3, shape, deep_out1, [self_simi_map1, other_simi_map1, entropy_weight1], \
            [self_simi_map2, other_simi_map2, entropy_weight2], [self_simi_map3, other_simi_map3,entropy_weight3]  # [stage_feat1[1],stage_feat2[1],stage_feat3[1]]  # ,deep_out1,deep_out2
        # return output1, output2, output3, deep_out1, [self_simi_map1, other_simi_map1, entropy_weight1], \
        #    [self_simi_map2, other_simi_map2, entropy_weight2], [self_simi_map3, other_simi_map3,entropy_weight3]
        # return output1, output2, output3, [self_simi_map1, other_simi_map1, entropy_weight1], \
        #    [self_simi_map2, other_simi_map2, entropy_weight2], [self_simi_map3, other_simi_map3,entropy_weight3]
class MCNet2d_v3(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v3, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 2,
                   'acti_func': 'relu'}
        params4 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 3,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        self.decoder4 = Decoder(params4)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        output4 = self.decoder4(feature)
        return output1, output2, output3, output4


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info

    model = UNet(in_chns=1, class_num=4).cuda()
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb;

    ipdb.set_trace()
