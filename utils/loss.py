# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import OKS_SIGMA
from utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from utils.torch_utils import autocast

from utils.metrics import bbox_iou, probiou
from utils.tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self,device, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        # device = next(model.parameters()).device  # get model device
        # h = model.args  # hyperparameters

        # m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.hyp = h
        self.stride = torch.tensor([8,16,32])#m.stride  # model strides
        self.nc = 1#m.nc  # number of classes
        self.no = 1+16*4#m.nc + m.reg_max * 4
        self.reg_max = 16#m.reg_max
        self.device = device

        self.use_dfl = 16 > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)# 生成 0 到 reg_max-1（reg_max 取值 16）的整数列表

    def preprocess(self, targets, batch_size, scale_tensor):#wr
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # scale_tensor [640,640,640,640]
        # targets [M+N+...,1+1+4=6] M+N+...代表一个batch里面bbox的数量总和，1+1+4表示1代表batch_idx(image index)，1代表label，4代表bbox
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j#判断 i 中的元素是否等于 j，并生成一个布尔掩码数组（Boolean mask）。
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out# (bs,max_count,5)

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution.
            anchor_points形状为(总网格数, 2)，其中总网格数等于所有特征图网格数之和.
            pred_distri (batch_size, num_anchors, reg_max * 4)
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 对最后一维（reg_max）做 softmax 转换成映射表 [0，15] 中每个整数的概率，再乘以映射表，得到均值，作为该锚点的某个偏移量的预估值。
            # pred_dist 变为 shape（batch, anchors, 4)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False) #(batch, anchors, 4)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        batch 为训练批次 dict，其中包括 GT 的 batch_idx, cls，bboxes 等属性"""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # # Detect 推理时返回(y,x)，训练时返回 x，故这里根据返回类型提取特征。
        # feats 为模型预测值，对于 Detect 任务，head.py 中 Detect 的输出为(x0,x1,x2)，x[0]的shape(N, reg_max*4+nc , H, W)
        feats = preds[1] if isinstance(preds, tuple) else preds # 推理和训练返回值不一样
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        # pred_distri b,reg_max * 4,-1
        # pred_scores b,nc,-1

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # pred_distri (batch_size, num_anchors, reg_max * 4)
        # pred_scores (batch_size, num_anchors, num_classes)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)=(80x80=640,640)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)    # 生成了用于检测任务的锚点坐标及其对应的步长
        # anchor_points形状为(总网格数, 2)，其中总网格数等于所有特征图网格数之和.
        # stride_tensor的形状为(总网格数, 1)，对应每个锚点的步长。

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) # [M+N+...,1+1+4]
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # imgsz[[1, 0, 1, 0]] 根据索引得到 [640,640,640,640]  #return  (bs,max_count,5)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls (bs,max_count,1), xyxy (bs,max_count,4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)   #(bs,max_count,1) 比较求和后的结果是否大于0.0，并将比较结果以布尔类型的形式存储在mask_gt中。

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4) (batch, anchors, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        """
            TaskAlignedAssigner()
           Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
            返回结果：返回标签、边界框、得分、前景掩码及真实框索引。
        """

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # 其中预测pred_scores: b x 8400 x cls_num (batch_size, num_anchors, num_classes);
        # target_scores: b x 8400 x cls_num,(bs, num_total_anchors, num_classes)
        # 相当于对于每个box，其cls_num个分类都视为二分类，并进行交叉熵运算。

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # loss[0] *= self.hyp.box  # box gain
        # loss[1] *= self.hyp.cls  # cls gain
        # loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, device):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(device)
        self.kpt_shape = [15, 3] # model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [15, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        # self.temporal = temporal

    def preprocess(self, targets, batch_size, scale_tensor):  # wr
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # scale_tensor [640,640,640,640]
        # targets [M+N+...,1+1+4=6] M+N+...代表一个batch里面bbox的数量总和，1+1+4表示1代表batch_idx(image index)，1代表label，4代表bbox
        # 1+1+1+4=7表示1代表batch_idx(image index)，1代表temporal_idx,1代表label， 4代表bbox
        #wr
        # batch_idx = targets[:, 0]
        # temporal_idx = targets[:, 1]
        # temporal_batch_idx = temporal_idx+batch_idx*self.temporal
        # targets[:, 0] = temporal_batch_idx

        # 下面使用与原代码基本相同处理逻辑
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size*self.temporal, 0, ne - 2, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size*self.temporal, counts.max(), ne - 2, device=self.device)
            for j in range(batch_size*self.temporal):
                matches = i == j  # 判断 i 中的元素是否等于 j，并生成一个布尔掩码数组（Boolean mask）。
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 2:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))#(bs,max_count,5)

        return out  # (batch_size*self.temporal,max_count,5)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]    # 推理和训练返回值不一样
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        # pred_distri (batch_size, num_anchors, reg_max * 4)
        # pred_scores (batch_size, num_anchors, num_classes)
        # pred_kpts (batch_size, num_anchors, kpts n*3)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)# 生成了用于检测任务的锚点坐标及其对应的步长
        # anchor_points形状为(总网格数, 2)，其中总网格数等于所有特征图网格数之和.
        # stride_tensor的形状为(总网格数, 1)，对应每个锚点的步长。

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx,batch["temporal_idx"].view(-1,1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)#wr
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  #return  (bs,max_count,5) # imgsz[[1, 0, 1, 0]] 根据索引得到 [640,640,640,640]
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4) # xyxy, (b, h*w, 4) (batch, anchors, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w.sum, 15, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        """
            Args:
                pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
                pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
                anc_points (Tensor): shape(num_total_anchors, 2)
                gt_labels (Tensor): shape(bs, n_max_boxes, 1)
                gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
                mask_gt (Tensor): shape(bs, n_max_boxes, 1)
    
            Returns:
                target_labels (Tensor): shape(bs, num_total_anchors)
                target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
                target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
                fg_mask (Tensor): shape(bs, num_total_anchors)
                target_gt_idx (Tensor): shape(bs, num_total_anchors)
                返回结果：返回标签、边界框、得分、前景掩码及真实框索引。
        """

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # 其中预测pred_scores: b x 8400 x cls_num (batch_size, num_anchors, num_classes);
        # target_scores: b x 8400 x cls_num,(bs, num_total_anchors, num_classes)
        # 相当于对于每个box，其cls_num个分类都视为二分类，并进行交叉熵运算。

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()  #shape [N, 15, 3] and format (x, y, visible).
            keypoints[..., 0] *= imgsz[1]#?
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= 7.5  #self.hyp.box  # box gain
        loss[1] *= 12.0 #self.hyp.pose  # pose gain
        loss[2] *= 1.0  #self.hyp.kobj  # kobj gain
        loss[3] *= 0.5  #self.hyp.cls  # cls gain
        loss[4] *= 1.5  #self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).#shape [N, 15, 3] and format (x, y, visible).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_bach, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_atnchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).# (b, h*w.sum, 15, 3)

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """

        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


if __name__ == "__main__":
    preds = torch.randn([1, 3, 640, 640])
    batch = torch.rand([1, 3, 640, 640])
    imgsz = torch.tensor(preds.shape[2:])
    print(imgsz)
    print(imgsz[[1, 0, 1, 0]])
    # criterion = v8PoseLoss()
    # loss =v8PoseLoss(preds, batch)