import argparse
import os
import shutil
import pandas as pd
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data2/tcg/csf-net/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='exp', help='experiment_name')
parser.add_argument('--model', type=str, default='mcnet2d_ours', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
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
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    test_save_path = "../model/ACDC/{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type='ours_net', in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path), strict=False)
    print("init weight from {}".format(save_model_path))
    net.eval()

    # first_total = 0.0
    # second_total = 0.0
    # third_total = 0.0
    # for case in tqdm(image_list):
    #     first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
    #     first_total += np.asarray(first_metric)
    #     second_total += np.asarray(second_metric)
    #     third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]

    # 初始化列表来存储每个样本的度量
    first_metrics = []
    second_metrics = []
    third_metrics = []

    # 遍历图像列表，收集度量
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_metrics.append(first_metric)
        second_metrics.append(second_metric)
        third_metrics.append(third_metric)

    # 将列表转换为numpy数组以便于计算
    first_metrics = np.array(first_metrics)
    second_metrics = np.array(second_metrics)
    third_metrics = np.array(third_metrics)

    first_metrics[:, 0] = first_metrics[:, 0] * 100
    second_metrics[:, 0] = second_metrics[:, 0] * 100
    third_metrics[:, 0] = third_metrics[:, 0] * 100

    # 计算均值
    avg_first_metric = np.mean(first_metrics,axis=0)
    avg_second_metric = np.mean(second_metrics,axis=0)
    avg_third_metric = np.mean(third_metrics,axis=0)


    # 计算方差
    std_first_metric = np.std(first_metrics,axis=0, ddof=1)  # ddof=1 表示样本方差
    std_second_metric = np.std(second_metrics,axis=0, ddof=1)
    std_third_metric = np.std(third_metrics,axis=0, ddof=1)
    # 创建一个DataFrame来存储所有的指标
    metrics_df = pd.DataFrame({
        'First Metric 1': first_metrics[:, 0],
        'First Metric 2': first_metrics[:, 1],
        'First Metric 3': first_metrics[:, 2],
        'First Metric 4': first_metrics[:, 3],
        'Second Metric 1': second_metrics[:, 0],
        'Second Metric 2': second_metrics[:, 1],
        'Second Metric 3': second_metrics[:, 2],
        'Second Metric 4': second_metrics[:, 3],
        'Third Metric 1': third_metrics[:, 0],
        'Third Metric 2': third_metrics[:, 1],
        'Third Metric 3': third_metrics[:, 2],
        'Third Metric 4': third_metrics[:, 3],
        # 'First Metric Mean': avg_first_metric,
        # 'Second Metric Mean': avg_second_metric,
        # 'Third Metric Mean': avg_third_metric,
        # 'First Metric Std': std_first_metric,
        # 'Second Metric Std': std_second_metric,
        # 'Third Metric Std': std_third_metric
    })

    # 将DataFrame写入CSV文件
    metrics_df.to_csv(test_save_path+'../metrics.csv', index=False)
    # 将结果打包
    avg_metrics = [avg_first_metric, avg_second_metric, avg_third_metric]
    var_metrics = [std_first_metric, std_second_metric, std_third_metric]

    # 返回均值和方差
    return avg_metrics, var_metrics,test_save_path
    # return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, std_metrics,test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
