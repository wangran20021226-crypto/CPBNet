import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image  = image.squeeze(0).cpu().detach().numpy()
    label  = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice_ = image[ind, :, :]
        x, y   = slice_.shape
        slice_ = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
        inp    = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().cuda()

        model.eval()
        with torch.no_grad():
            # UNet v8 返回 6 个值，只取前两个分割输出
            outs   = model(inp)
            output = (outs[0] + outs[1]) / 2
            out    = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out    = out.cpu().detach().numpy()
            pred   = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_cross(image, label, model_l, model_r,
                             classes, patch_size=[256, 256]):
    image  = image.squeeze(0).cpu().detach().numpy()
    label  = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice_ = image[ind, :, :]
        x, y   = slice_.shape
        slice_ = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
        inp    = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().cuda()

        model_l.eval()
        model_r.eval()
        with torch.no_grad():
            out_l  = model_l(inp)[0]
            out_r  = model_r(inp)[0]
            output = (out_l + out_r) / 2
            out    = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out    = out.cpu().detach().numpy()
            pred   = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list