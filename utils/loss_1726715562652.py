import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import OKS_SIGMA
from utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from utils.torch_utils import autocast

from utils.metrics import bbox_iou, probiou
from utils.tal import bbox2dist

def __call__(self, preds, batch):
    """
    计算框、类别和dfl损失之和乘以批次大小。

    参数:
    preds -- 模型预测，可能是元组或单个张量。
    batch -- 包含目标信息的批次数据。

    返回:
    损失的总和乘以批次大小，以及分离的损失项(box, cls, dfl)。
    """
    # 初始化损失张量，设备与输入设备相同
    loss = torch.zeros(3, device=self.device)  # 分别对应box, cls, dfl损失

    # 检查预测是否为元组，否则直接赋值
    feats = preds[1] if isinstance(preds, tuple) else preds

    # 将多尺度特征图整合并分割成分布和分数
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        (self.reg_max * 4, self.nc), 1
    )
    # pred_distri: (batch_size, reg_max * 4, -1)
    # pred_scores: (batch_size, nc, -1)

    # 调整维度顺序，以便后续计算
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    # pred_distri: (batch_size, num_anchors, reg_max * 4)
    # pred_scores: (batch_size, num_anchors, num_classes)

    # 定义数据类型和批次大小，计算图像尺寸
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像尺寸

    # 生成锚点和步长张量
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    # 处理目标数据
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # 分割出标签和框

    # 筛选有效目标
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    # 解码预测的框
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # 预测的框
    # pred_bboxes: (batch_size, num_anchors, 4)

    # 分配目标
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    # 计算目标分数总和，避免除以零
    target_scores_sum = max(target_scores.sum(), 1)

    # 计算分类损失
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
    # 其中预测pred_scores: b x 8400 x cls_num; target_scores: b x 8400 x cls_num,
    # 相当于对于每个box，其cls_num个分类都视为二分类，并进行交叉熵运算。

    # 计算框损失
    if fg_mask.sum():
        target_bboxes /= stride_tensor
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )

    # 根据超参数调整损失权重
    loss[0] *= self.hyp.box  # box损失增益
    loss[1] *= self.hyp.cls  # 类别损失增益
    loss[2] *= self.hyp.dfl  # 分布损失增益

    # 返回总损失和单独的损失项
    return loss.sum() * batch_size, loss.detach()
