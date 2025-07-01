import torch
from torch import nn
import sys
import numpy as np
import torch.nn.functional as F

sys.path.append("/data2/tcg/csf-net/code/vmamba")
from vmamba.SEEM import *

def masked_average_pooling(feature, mask, reliability_map=None):
    # print(feature.shape[-2:])
    if torch.is_tensor(reliability_map):
        mask = F.interpolate(mask, size=feature.shape[-3:], mode='trilinear', align_corners=True)
        mask = mask*reliability_map[:,None,:,:]
        # print((feature*mask).shape)
        masked_feature = torch.sum(feature * mask, dim=(2, 3,4))
    else:
        mask = F.interpolate(mask, size=feature.shape[-3:], mode='trilinear', align_corners=True)
        # print((feature*mask).shape)
        masked_feature = torch.sum(feature * mask, dim=(2, 3,4)) / (mask.sum(dim=(2, 3,4)) + 1e-5)
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
    gt_onthot = F.one_hot(score_map, 2)
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
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
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
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)
        self.mambablock = ResidualBlock(input_dim, input_dim * 2, 1024, 16, 2, dt_rank=None, d_conv=3,
                                        pad_vocab_size_multiple=8, conv_bias=True, bias=False)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)
            feature = feature.permute(1, 0, 2).contiguous()
            # feature = self.encoder_layers(feature)
            feature = self.mambablock(feature)
            feature = feature.permute(1, 0, 2).contiguous()
            feature = feature.view(b, h, w,d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature


class IntraSampleAttention(torch.nn.Module):
    """
    Implementation for intra-sample self-attention
    input size for the encoder_layers: [h x w x d, batch, dim]
    """

    def __init__(self, input_dim=256, hidden_dim=1024):
        super(IntraSampleAttention, self).__init__()
        self.input_dim = input_dim
        self.encoder_layers = torch.nn.TransformerEncoderLayer(input_dim, 4, hidden_dim, 0.5)
        self.mambablock = ResidualBlock(input_dim, input_dim * 2, 1024, 16, 2, dt_rank=None, d_conv=3,
                                        pad_vocab_size_multiple=8, conv_bias=True, bias=False)

    def forward(self, feature):
        if self.training:
            b, c, h, w, d = feature.shape
            feature = feature.permute(0, 2, 3, 4, 1).contiguous()
            feature = feature.view(b, h * w * d, c)  # （24，1024,128）
            # feature = feature.permute(1, 0, 2).contiguous()
            feature = self.mambablock(feature)
            # feature = self.encoder_layers(feature)
            # feature = feature.permute(1, 0, 2).contiguous()
            feature = feature.view(b, h, w, d, c)
            feature = feature.permute(0, 4, 1, 2, 3).contiguous()
        return feature

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv3d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm3d(out_chn),
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
            torch.nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv3d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm3d(out_chn),
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
                torch.nn.Conv3d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                nn.BatchNorm3d(dim_in // 2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim_out),
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


class SideConv(nn.Module):
    def __init__(self, n_classes=4):
        super(SideConv, self).__init__()

        self.side4 = nn.Conv3d(128, n_classes, 1, padding=0)
        self.side3 = nn.Conv3d(64, n_classes, 1, padding=0)
        self.side2 = nn.Conv3d(32, n_classes, 1, padding=0)
        self.side1 = nn.Conv3d(32, n_classes, 1, padding=0)
        # self.side1 = nn.Conv2d(16, n_classes, 1, padding=0)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

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


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters*2, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters*2, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class EncoderAuxiliary(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(EncoderAuxiliary, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters*2, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters*2, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.intra_attention = IntraSampleAttention(self.block_four.conv[6].weight.shape[0],
                                                    self.block_four.conv[6].weight.shape[0] * 4)
        self.inter_attention = InterSampleAttention(self.block_four.conv[6].weight.shape[0],
                                                    self.block_four.conv[6].weight.shape[0] * 4)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)

        x4 = self.intra_attention(x4) + x4
        x4 = self.inter_attention(x4) + x4

        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder_pro(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder_pro, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters*2, normalization=normalization,
                                                  mode_upsampling=up_type)
        # self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
        #                                           mode_upsampling=up_type)
        self.block_nine = convBlock(1, n_filters*2, n_filters*2, normalization=normalization)
        # self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters*2, n_classes, 1, padding=0)
        # self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)
        output = self.out_conv(x9)

        mask = torch.softmax(output, dim=1)
        uncertainty = -torch.sum(mask * torch.log(mask + 1e-16), dim=1)
        norm_uncertainty = torch.stack([uncertain / torch.sum(uncertain) for uncertain in uncertainty], dim=0)
        reliability_map = (1 - norm_uncertainty) / np.prod(np.array(norm_uncertainty.shape[-3:]))
        batch_pro = batch_prototype(x9, mask, reliability_map)
        # similarity_map = adjusted_cosine_similarity(x8, batch_pro)
        similarity_map = similarity_calulation(x9, batch_pro)
        entropy_weight = agreementmap(similarity_map)
        self_simi_map = selfsimilaritygen(similarity_map)  # B*HW*C
        other_simi_map = othersimilaritygen(similarity_map)  # B*HW*C

        return output, self_simi_map, other_simi_map, entropy_weight, [x6, x7, x8, x9]

    
class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters*2, normalization=normalization, mode_upsampling=up_type)
        # self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters*2, n_filters*2, normalization=normalization)
        # self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters*2, n_classes, 1, padding=0)
        # self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        
        return out_seg
 
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1

class MCNet3d_v1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2
    
class MCNet3d_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        out_seg3 = self.decoder3(features)
        return out_seg1, out_seg2, out_seg3


class ours_net_3d(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(ours_net_3d, self).__init__()

        self.encoder1 = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.encoder2 = EncoderAuxiliary(n_channels +2, n_classes, n_filters, normalization, has_dropout, has_residual)
        # self.encoder2 = EncoderAuxiliary(n_channels +2, n_classes, n_filters, normalization, has_dropout, has_residual)
        # self.decoder1 = Decoder_pro(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        # self.decoder2 = Decoder_pro(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        # self.decoder3 = Decoder_pro(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)
        self.DF_Module = DF_Module(n_classes, n_classes, True).cuda()
        self.sideconv1 = SideConv(n_classes=n_classes)

    def forward(self, input):
        features1 = self.encoder1(input)
        # output1, self_simi_map1, other_simi_map1, entropy_weight1, stage_feat1 = self.decoder1(features1)
        # output2, self_simi_map2, other_simi_map2, entropy_weight2, stage_feat2 = self.decoder2(features1)
        output1 = self.decoder1(features1)
        output2 = self.decoder2(features1)
        shape_ori = self.DF_Module(output1, output2)
        shape = F.softmax(shape_ori, dim=1)
        # deep_out1 = self.sideconv1(stage_feat1)
        shape_input = torch.cat((input, shape), dim=1)

        feature2 = self.encoder2(shape_input)
        # output3, self_simi_map3, other_simi_map3, entropy_weight3, stage_feat3 = self.decoder3(feature2)
        output3 = self.decoder3(feature2)

        return output1, output2, output3, shape_ori
        # return output1, output2, output3, shape_ori, deep_out1, [self_simi_map1, other_simi_map1, entropy_weight1], \
        #     [self_simi_map2, other_simi_map2, entropy_weight2], [self_simi_map3, other_simi_map3,entropy_weight3]

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
