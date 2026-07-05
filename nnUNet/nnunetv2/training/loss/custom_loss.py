import torch
from torch import nn
import torch.nn.functional as F
import  numpy as np
from  nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

from PIL import Image
import os

# def save_skeleton_frames(skel_pred, skel_true, save_root="skeleton_frames"):
#     """
#     保存骨架Tensor的每一层depth为图片到指定目录
#
#     Args:
#         skel_pred (torch.Tensor): 预测骨架，形状 [2, 1, 128, 128, 128]
#         skel_true (torch.Tensor): 真实骨架，形状 [2, 1, 128, 128, 128]
#         save_root (str): 保存根目录，会自动创建子目录区分预测/真实
#     """
#     # 1. 创建保存目录（自动创建不存在的目录）
#     pred_dir = os.path.join(save_root, "pred_skeleton")
#     true_dir = os.path.join(save_root, "true_skeleton")
#     os.makedirs(pred_dir, exist_ok=True)
#     os.makedirs(true_dir, exist_ok=True)
#
#     # 2. Tensor预处理：去除通道维度 + 归一化到[0,255] + 转为numpy
#     def preprocess(tensor):
#         # 去除通道维度（shape从[2,1,128,128,128] → [2,128,128,128]）
#         tensor = tensor.squeeze(1)  # 挤压第1维（channel维）
#         # 归一化到[0, 255]（处理预测值可能的概率范围或真实值的0/1）
#         tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # 避免除零
#         tensor = (tensor * 255).byte()  # 转为uint8类型
#         return tensor.cpu().numpy()  # 转移到CPU并转为numpy数组（支持PIL读取）
#
#     skel_pred_np = preprocess(skel_pred)
#     skel_true_np = preprocess(skel_true)
#
#     # 3. 遍历batch和depth层，保存图片
#     batch_size = skel_pred_np.shape[0]  # batch_size=2
#     depth_num = skel_pred_np.shape[1]  # depth_num=128
#
#     for batch_idx in range(batch_size):
#         for depth_idx in range(depth_num):
#             # 读取当前batch、当前depth层的图像（shape: [128, 128]）
#             pred_img = skel_pred_np[batch_idx, depth_idx, :, :]
#             true_img = skel_true_np[batch_idx, depth_idx, :, :]
#
#             # 图片命名规则：batch_{索引}_depth_{索引}.png（索引从0开始）
#             img_name = f"batch_{batch_idx}_depth_{depth_idx:03d}.png"  # depth用3位数字（000-127）
#
#             # 保存预测骨架图片
#             pred_path = os.path.join(pred_dir, img_name)
#             Image.fromarray(pred_img).save(pred_path)
#
#             # 保存真实骨架图片
#             true_path = os.path.join(true_dir, img_name)
#             Image.fromarray(true_img).save(true_path)
#
#     print(f"保存完成！\n预测骨架：{pred_dir}\n真实骨架：{true_dir}")





class DC_and_CE_and_CLDice_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_cl_dice=1, smooth=1., num_iter=60, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_and_CLDice_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.smooth = smooth
        self.weight_cl_dice = weight_cl_dice

        self.soft_skeletonize = SoftSkeletonize(num_iter=num_iter)

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """


        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        skel_pred = self.soft_skeletonize(net_output[:, 1:, :, :])
        skel_true = self.soft_skeletonize(target)
        tprec = (torch.sum(torch.multiply(skel_pred, target))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, net_output))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

    
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cl_dice * cl_dice
        return result


class SoftCLDice_loss(nn.Module):
    def __init__(self, smooth=1., num_iter=60):

        super(SoftCLDice_loss, self).__init__()

        self.smooth = smooth

        self.soft_skeletonize = SoftSkeletonize(num_iter=num_iter)


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        skel_pred = self.soft_skeletonize(net_output[:, 1:, :, :])
        skel_true = self.soft_skeletonize(target)
        tprec = (torch.sum(torch.multiply(skel_pred, target))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, net_output))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)

        # save_skeleton_frames(skel_pred, skel_true, save_root="path/to/tmp")

        result = cl_dice
        return result


class DSHLoss(nn.Module):
    def __init__(self, alpha=0.99, smooth=1e-5, num_iter=40, distance_iter=16):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.distance_iter = distance_iter
        self.soft_skeletonize = SoftSkeletonize(num_iter=num_iter)

    @staticmethod
    def _foreground_probability(net_output):
        if net_output.shape[1] == 1:
            return torch.sigmoid(net_output)
        return F.softmax(net_output, dim=1)[:, 1:2]

    def _soft_dice_loss(self, pred, target):
        axes = tuple(range(2, pred.ndim))
        numerator = 2 * torch.sum(pred * target, dim=axes)
        denominator = torch.sum(pred.square(), dim=axes) + torch.sum(target.square(), dim=axes)
        dice = (numerator + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()

    def _approx_distance_map(self, mask):
        with torch.no_grad():
            mask = (mask > 0.5).float()
            reached = mask
            seen = mask
            dist = torch.full_like(mask, float(self.distance_iter))
            dist = torch.where(mask > 0, torch.zeros_like(dist), dist)
            for step in range(1, self.distance_iter + 1):
                reached = F.max_pool3d(reached, kernel_size=3, stride=1, padding=1)
                new = (reached > 0.5).float() * (1 - seen)
                dist = torch.where(new > 0, torch.full_like(dist, float(step)), dist)
                seen = torch.maximum(seen, new)
            return dist / float(self.distance_iter)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        target = (target > 0).float()
        pred = self._foreground_probability(net_output)
        dice_loss = self._soft_dice_loss(pred, target)

        skel_pred = self.soft_skeletonize(pred)
        skel_true = self.soft_skeletonize(target)
        dist_true = self._approx_distance_map(skel_true)
        dist_pred = self._approx_distance_map(skel_pred.detach())
        hd_loss = (((skel_pred - skel_true).square()) * (dist_true.square() + dist_pred.square())).mean()
        return self.alpha * dice_loss + (1 - self.alpha) * hd_loss

class LPE_theta1_loss(nn.Module):
    def forward(self, p_y_condi_x, p_y_condi_sx):
        loss = F.binary_cross_entropy_with_logits(p_y_condi_x, p_y_condi_sx)
        return loss.mean()


class LPE_theta2_loss(nn.Module):
    def forward(self, p_s_condi_yx, s, p_y_condi_sx):
        s = s.to(dtype=p_s_condi_yx.dtype, device=p_s_condi_yx.device)

        loss = F.binary_cross_entropy_with_logits(p_s_condi_yx, s, weight=p_y_condi_sx)
        return loss.mean()

softmax_helper = lambda x: F.softmax(x, 1)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class Tversky_loss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-6, alpha=0.1):
        """
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp + self.smooth
        denominator = tp + self.alpha * fp + (1 - self.alpha) *fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc



class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        img = img.float()

        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        img = img.float()

        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):

        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):


        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel.clamp(0, 1)

    def forward(self, img):

        return self.soft_skel(img)

