import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.Generate_Prototype import *
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, val_2d
from utils.losses import to_one_hot

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data2/tcg/csf-net/data/ACDC/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='exp', help='experiment_name')
parser.add_argument('--model', type=str, default='ours_net', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
# costs
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--beta', type=float, default=1, help='weight to balance all losses')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate"in dataset:
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    # elif "Prostate":
    #     ref_dict = {"2": 27, "4": 53, "8": 120,
    #                 "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def prototype_pred(src_prototypes, feature_tgt, mask_src, class_nums):
    # 1 extract the foreground features via masked average pooling
    feature_tgt_adj = F.interpolate(feature_tgt, size=mask_src.shape[-3:], mode='bilinear')  # 3D uses tri, 2D uses bilinear, [2, 256, 96, 96, 96]
    # print('feature_tgt_adj:', feature_tgt_adj.shape)
    for class_index in range(class_nums):
        dist = calDist(feature_tgt_adj, src_prototypes[class_index]).unsqueeze(1)
        final_dist = dist if class_index == 0 else torch.cat((final_dist, dist), 1)
    final_dist_soft = torch.softmax(final_dist, dim=1)
    return final_dist_soft

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    consistency_criterion = losses.mse_loss

    dice_loss = losses.DiceLoss(num_classes)
    self_proloss = losses.weight_self_pro_softmax_mse_loss
    cross_proloss = losses.double_weight_cross_pro_softmax_mse_loss  # (weight,input_logits, target_logits)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance_shape = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs = model(volume_batch)
            num_outputs = len(outputs)

            y_ori = torch.zeros((3,) + outputs[0].shape)
            y_pseudo_label = torch.zeros((3,) + outputs[0].shape)

            loss_seg = 0
            loss_seg_dice = 0
            # fts_confidence = []
            for idx in range(num_outputs):
                if (idx < 3):
                    y = outputs[idx][:labeled_bs, ...]
                    y_prob = F.softmax(y, dim=1)
                    loss_seg += ce_loss(y, label_batch[:labeled_bs][:].long())
                    loss_seg_dice += dice_loss(y_prob, label_batch[:labeled_bs].unsqueeze(1))

                    y_all = outputs[idx]
                    y_prob_all = F.softmax(y_all, dim=1)
                    y_ori[idx] = y_prob_all
                    y_pseudo_label[idx] = sharpening(y_prob_all)
                ##########################################消融实验
                if (idx == 3):
                    y = outputs[idx][:labeled_bs, ...]
                    y_prob = F.softmax(y, dim=1)
                    loss_seg += ce_loss(y, label_batch[:labeled_bs][:].long())
                    loss_seg_dice += args.beta * dice_loss(y_prob, label_batch[:labeled_bs].unsqueeze(1))
                if (idx == 4):
                ###########################################
                # if (idx == 3):
                    length = len(outputs[idx])
                    los_weight = torch.tensor([0.8, 0.6, 0.4, 0.2], dtype=torch.float32).cuda()
                    for i in range(length):
                        y = outputs[idx][i][:labeled_bs, ...]
                        y_prob = F.softmax(y, dim=1)
                        loss_seg += ce_loss(y, label_batch[:labeled_bs][:].long()) * los_weight[i]
                        loss_seg_dice += dice_loss(y_prob, label_batch[:labeled_bs].unsqueeze(1)) * los_weight[i]
            loss_consist = 0
            for i in range(3):
                for j in range(3):
                    if i != j:
                        loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
##########################################################################################################################

            selfproout1 = outputs[5][0]
            selfproout2 = outputs[6][0]
            selfproout3 = outputs[7][0]

            crossproout1 = outputs[5][1]
            crossproout2 = outputs[6][1]
            crossproout3 = outputs[7][1]

            entropy1 = outputs[5][2]
            entropy2 = outputs[6][2]
            entropy3 = outputs[7][2]


            consistency_self_pro1 = (self_proloss(selfproout1, outputs[1], entropy1) +self_proloss(selfproout1, outputs[2], entropy1))/2
            consistency_cross_pro1 = (cross_proloss(selfproout1, crossproout1, outputs[1], entropy1) +cross_proloss(selfproout1, crossproout1, outputs[2], entropy1))/2

            consistency_self_pro2 = (self_proloss(selfproout2, outputs[0], entropy2)+self_proloss(selfproout2, outputs[2], entropy2))/2
            consistency_cross_pro2 = (cross_proloss(selfproout2, crossproout2, outputs[0], entropy2)+cross_proloss(selfproout2, crossproout2, outputs[2], entropy2))

            consistency_self_pro3 = (self_proloss(selfproout3, outputs[0], entropy3)+self_proloss(selfproout3, outputs[1], entropy3))/2
            consistency_cross_pro3 = (cross_proloss(selfproout3, crossproout3, outputs[0], entropy3)+cross_proloss(selfproout3, crossproout3, outputs[1], entropy3))/2

            consistency_loss_aux1 = torch.mean(consistency_self_pro1) + torch.mean(consistency_self_pro2) + torch.mean(consistency_self_pro3)
            consistency_loss_aux2 = torch.mean(consistency_cross_pro1) + torch.mean(consistency_cross_pro2) + torch.mean(consistency_cross_pro3)
            consistency_loss1 = (consistency_loss_aux1 + consistency_loss_aux2) / 2

            #
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            #

##########################################################################################################################

            loss = args.lamda * loss_seg_dice + consistency_weight * (loss_consist + consistency_loss1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' % (iter_num, loss, loss_seg_dice, loss_consist))
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_consist', loss_consist, iter_num)
            # writer.add_scalar('loss/consistency_loss_aux1', consistency_loss_aux1, iter_num)
            # writer.add_scalar('loss/consistency_loss_aux2', consistency_loss_aux2, iter_num)
            # writer.add_scalar('loss/loss_pt', loss_pt, iter_num)
            # writer.add_scalar('loss/loss_pc_lab', loss_pc_lab, iter_num)
            # writer.add_scalar('loss/loss_pc_unlab', loss_pc_unlab, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # if iter_num % 2 == 0:
            if epoch > 0 and epoch % 5 == 0:
            # if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                # metric_list_shape = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    # metric_i_shape = val_2d.test_single_volume_shape(sampled_batch["image"], sampled_batch["label"], model,
                    #                                      classes=num_classes)
                    metric_list += np.array(metric_i)
                    # metric_list_shape += np.array(metric_i_shape)
                metric_list = metric_list / len(db_val)
                # metric_list_shape = metric_list_shape / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)
                    # writer.add_scalar('info/val_shape_{}_dice'.format(class_i + 1), metric_list_shape[class_i, 0], iter_num)
                    # writer.add_scalar('info/val_shape_{}_hd95'.format(class_i + 1), metric_list_shape[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                # performance_shape = np.mean(metric_list_shape, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                # mean_hd95_shape = np.mean(metric_list_shape, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                # writer.add_scalar('info/val_shape_mean_dice', performance_shape, iter_num)
                # writer.add_scalar('info/val_shape_mean_hd95', mean_hd95_shape, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                # if performance_shape > best_performance_shape:
                #     best_performance_shape = performance_shape
                #     save_mode_path = os.path.join(snapshot_path,
                #                                   'iter_{}_dice_shape_{}.pth'.format(iter_num, round(best_performance_shape, 4)))
                #     save_best_path = os.path.join(snapshot_path, '{}_shape_best_model.pth'.format(args.model))
                #     torch.save(model.state_dict(), save_mode_path)
                #     torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                # logging.info('iteration %d : mean_shape_dice : %f mean_shape_hd95 : %f' % (iter_num, performance_shape, mean_hd95_shape))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/ACDC/{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
