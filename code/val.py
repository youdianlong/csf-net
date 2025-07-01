import argparse
import os
import shutil
import sys
import re
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
sys.path.append("/data2/tcg/MC-Net-ours/code/")
from networks.net_factory import net_factory


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data2/tcg/csf-net/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='exp', help='experiment_name')
parser.add_argument('--model', type=str, default='ours_net', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--valroot_path', type=str, default='path', help='Name of Experiment')



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main) > 1:
                out_main = out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1) == 0:
        first_metric = 0, 0, 0, 0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2) == 0:
        second_metric = 0, 0, 0, 0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3) == 0:
        third_metric = 0, 0, 0, 0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    return first_metric, second_metric, third_metric


def Inference(FLAGS,mod=None):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    net = net_factory(net_type='ours_net', in_chns=1, class_num=FLAGS.num_classes)

    save_model_path = FLAGS.valroot_path +'/'+ mod.strip()
    net.load_state_dict(torch.load(save_model_path), strict=False)
    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    val_path = FLAGS.valroot_path
    file_names = []
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    for file_name in os.listdir(val_path):
        if file_name.endswith(".pth"):
            match = re.search(r'dice_([0-9.]+)', file_name)
            if match:
                dice_score_str = match.group(1)
                dice_score_str = ''.join(filter(str.isdigit or str.isdecimal, dice_score_str))
                try:
                    dice_score = float(dice_score_str)
                    if dice_score > 8600:
                        file_names.append(file_name)
                except ValueError:
                    print('error')
    val_list_path = os.path.join(val_path, 'valname.list')
    with open(val_list_path, 'w') as f:
        for name in file_names:
            f.write(name + '\n')
    with open(FLAGS.valroot_path + '/valname.list', 'r') as f:
        valname_list = f.readlines()

    max = 0
    for mod in valname_list:
        metrics = Inference(FLAGS,mod)
        mean=(metrics[0] + metrics[1] + metrics[2]) / 3
        results_path = os.path.join(val_path, 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f'Metrics: {mean}\n')
            f.write(f'Mean Metric: {mod}\n')

        if max<mean[0]:
            max=mean[0]
            print(metrics)
            print((metrics[0] + metrics[1] + metrics[2]) / 3)
            print(mod)