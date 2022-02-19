# metrics.py

import torch
import torch.nn as nn
import numpy as np


"""
True Positive （真正， TP）预测为正的正样本
True Negative（真负 , TN）预测为负的负样本 
False Positive （假正， FP）预测为正的负样本
False Negative（假负 , FN）预测为负的正样本
"""


def diceCoeff(pred, gt, smooth=1e-5, ):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    score = (2 * intersection + smooth) / (unionset + smooth)

    return score.sum() / N


def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() /N


def diceCoeffv3(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    # 转为float，以防long类型之间相除结果为0
    score = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

    return score.sum()/N


def jaccard(pred, gt):
    """TP / (TP + FP + FN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = tp.float() / (tp + fp + fn).float()
    return score.sum()


def tversky(pred, gt, eps=1e-5,  alpha=0.8):
    """TP / (TP + (1-alpha) * FP + alpha * FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (tp + eps) / (tp + (1-alpha) * fp + alpha*fn + eps)
    return score.sum()/N


def accuracy(pred, gt):
    """(TP + TN) / (TP + FP + FN + TN)"""

    N = gt.size(0)
    pred_flat = pred.contiguous().view(N, -1)
    gt_flat = gt.contiguous().view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp + tn).float() / (tp + fp + tn + fn).float()

    return score.sum()


def precision(pred, gt):
    """TP / (TP + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.contiguous().view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))

    score = tp.float() / (tp + fp).float()

    return score.sum()


def sensitivity(pred, gt):
    """TP / (TP + FN)"""
    N = gt.size(0)
    pred_flat = pred.contiguous().view(N, -1)
    gt_flat = gt.contiguous().view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = tp.float() / (tp +  fn).float()

    return score.sum()


def specificity(pred, gt):
    """TN / (TN + FP)"""

    N = gt.size(0)
    pred_flat = pred.contiguous().view(N, -1)
    gt_flat = gt.contiguous().view(N, -1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))

    score = tn.float() / (fp + tn).float()

    return score.sum()


def recall(pred, gt):

    return sensitivity(pred, gt)


if __name__ == '__main__':

    # shape = torch.Size([2, 3, 4, 4])
    # 模拟batch_size = 2
    '''
    1 0 0= bladder
    0 1 0 = tumor
    0 0 1= background 
    '''


    pred = torch.Tensor([[
        [[1, 1, 1, 1],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])

    gt = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])


    dice1 = diceCoeff(pred[:, 0:1, :], gt[:, 0:1, :],)
    dice2 = jaccard(pred[:, 0:1, :], gt[:, 0:1, :], )
    dice3 = diceCoeffv3(pred[:, 0:1, :], gt[:, 0:1, :], )
    dice4 = diceCoeffv3(pred[:, 0:1, :], gt[:, 0:1, :], )

    accuracy = accuracy(pred[:, 0:1, :], gt[:, 0:1, :],  )
    recall = recall(pred[:, 0:1, :], gt[:, 0:1, :], )

    precision = precision(pred[:, 0:1, :], gt[:, 0:1, :], )

    sensitivity = sensitivity(pred[:, 0:1, :], gt[:, 0:1, :], )
    specificity = specificity(pred[:, 0:1, :], gt[:, 0:1, :], )
    tversky = tversky(pred[:, 0:1, :], gt[:, 0:1, :], )
    print(dice1, dice2, dice3,dice4)
    print(accuracy,recall)
    print(precision)
    print(sensitivity,specificity)
    print(tversky)

#otice: tp/fp/(dim=1) (have no /N) ;  matrix(have /N)
