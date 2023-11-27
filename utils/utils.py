from __future__ import print_function
import os
import shutil
import torch
import numpy as np

def save_checkpoint(state, is_best, root, filename):
    """Saves the model checkpoint and, if it is the best model, saves a copy with a specific naming convention."""
    checkpoint_path = os.path.join(root, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(root, 'best_' + filename))

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all metrics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the metrics with a new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_seg(pred, gt):
    """Evaluates segmentation metrics based on predictions and ground truths."""
    pred_binary = (pred >= 0.5).float().cuda()
    pred_binary_inverse = (pred_binary == 0).float().cuda()

    gt_binary = (gt >= 0.5).float().cuda()
    gt_binary_inverse = (gt_binary == 0).float().cuda()

    # Calculation of metrics
    MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
    TP = pred_binary.mul(gt_binary).sum().cuda(0)
    FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
    FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

    TP = torch.Tensor([1]).cuda(0) if TP.item() == 0 else TP
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Dice = 2 * Precision * Recall / (Precision + Recall)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    IoU_polyp = TP / (TP + FP + FN)

    # Returning metrics
    return (MAE.data.cpu().numpy().squeeze(), Recall.data.cpu().numpy().squeeze(),
            Precision.data.cpu().numpy().squeeze(), Accuracy.data.cpu().numpy().squeeze(),
            Dice.data.cpu().numpy().squeeze(), IoU_polyp.data.cpu().numpy().squeeze())

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup for consistency weight."""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=5, start_epoch=0):
    """Calculates the current weight for consistency, increasing it over epochs."""
    if epoch <= start_epoch:
        return 0.0
    return consistency * sigmoid_rampup(epoch - start_epoch, consistency_rampup - start_epoch)
