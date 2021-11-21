import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction='none', loss_type='iou'):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))  # 计算重合框左上
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))  # 计算重合框右下

        area_p = torch.prod(pred[:, 2:], 1)  # 预测框面积
        area_g = torch.prod(target[:, 2:], 1)  # 真值框面积

        en = (tl < br).type(tl.type()).prod(dim=1)  # 返回两框是否相交
        area_i = torch.prod(br - tl, 1) * en  # 相交的会返回大小，未相交的返回零
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)  # 得到iou

        if self.loss_type == 'iou':
            loss = 1 - iou ** 2
        elif self.loss_type == 'qiou':
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))  # 计算大框左上
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))  # 计算大框右下
            area_c = torch.prod(c_br - c_tl, 1)  # 大框面积
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)  # 截断下届
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class YOLOLoss(nn.Module):
    def __init__(self, num_class, strides=[8, 16, 32]):
        super().__init__()
        self.num_class = num_class
        self.strides = strides
        self.bcewithlof_less = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = IOUloss(reduction='none')
        self.grids = [torch.zeros(1)] * len(strides)  # 得到输出的数目个网格

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]  # 取出一个网格
        hsize, wsize = output.shape[-2:]  # 得到这个网格的尺寸
        if grid.shape[2:4] != output.shape[-2:]:  # 在网格尺寸与输入尺寸匹配时
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])  # 创建两个长宽与网格尺寸相同的矩阵
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())  # 拼接矩阵，组成[1,hsize,wsize,2]的形状
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)  # 矩阵尺寸变成[1,hsize*wsize,2]的形状，第三维网格代表他是处于x，y的第几个位置

        output = output.flatten(start_dim=2).permute(0, 2, 1)
        # output尺寸从[batch_size，num_class+5,20/40/80,20/40/80]变为[batch_size,400/1600/6400,num_class+5]
        output[..., :2] = (output[..., :2] + grid) * stride
        # 从相对所在方格点右上的距离转换成了相对整张图右上的距离
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        # 也把长宽恢复了处理
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        bbox_preds = outputs[:, :, :4]  # 尺寸为[batch, 预选框数目, 4],表示预选框
        obj_preds = outputs[:, :, 4:5]  # 尺寸为[batch, 预选框数目, 1],表示目标置信度
        cls_preds = outputs[:, :, 5:]  # 尺寸为[batch, 预选框数目, 目标种类数],表示目标总种类
        total_num_anchors = outputs.shape[1]  # 预选框数目

        x_shifts = torch.cat(x_shifts, 1)  # 变换尺寸
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []  # 设置相应目标
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx]) #取出对应标签
            if num_gt == 0: #如果标签不存在
                cls_target = outputs.new_zeros((0, self.num_classes))   #目标全部置零
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:   #标签存在时
                gt_bboxes_per_image = labels[batch_idx][..., :4]    #取出真值框坐标
                gt_classes = labels[batch_idx][..., 4]              #取出正确类别
                bboxes_preds_per_image = bbox_preds[batch_idx]      #取出预测框位置
                cls_preds_per_image = cls_preds[batch_idx]          #预测框种类
                obj_preds_per_image = obj_preds[batch_idx]          #预测框置信度

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg

    def forward(self, inputs, labels=None):  # input尺寸为[batch_size,num_class+5,20/40/80,20/40/80]若干方格下的位置+类别
        outputs = []  # 尺寸为[batch_size,400/1600/6400,num_class+5]若干方格下的位置+类别损失
        x_shifts = []  # 尺寸为[batch_size,400/1600/6400]
        y_shifts = []  # 尺寸为[batch_size,400/1600/6400]
        expanded_strides = []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])  # 记录x的偏移方格数
            y_shifts.append(grid[:, :, 1])  # 记录y的偏移方格数
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)  # 记录转化为图片根据像素的位置

        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1))
