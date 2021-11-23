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
            num_gt = len(labels[batch_idx])  # 取出对应标签
            if num_gt == 0:  # 如果标签不存在，图片上没有真值框
                cls_target = outputs.new_zeros((0, self.num_classes))  # 目标全部置零
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:  # 标签存在时
                gt_bboxes_per_image = labels[batch_idx][..., :4]  # 取出真值框坐标
                gt_classes = labels[batch_idx][..., 4]  # 取出正确类别
                bboxes_preds_per_image = bbox_preds[batch_idx]  # 取出预测框位置
                cls_preds_per_image = cls_preds[batch_idx]  # 预测框种类
                obj_preds_per_image = obj_preds[batch_idx]  # 预测框置信度

                #对应的真值类，对应的预选框，正样本预选框与真值类的iou矩阵，匹配的真值框，预选框数量
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts,
                )
                torch.cuda.empty_cache()
                num_fg += num_fg_img    #正特征点数量总和
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)#得到一个[成功匹配真值框数，种类数]，数值为iou独热
                obj_target = fg_mask.unsqueeze(-1)#得到一个[预选框总数，1],里面被选为正样本的部分置1
                reg_target = gt_bboxes_per_image[matched_gt_inds]#得到一个[预选框数，4],保存成功匹配的真值框中的信息
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)  #[图片数，预选框数，4]
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)        #[图片数，预选框总数]
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0) #[图片数*预选框数，4]
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)   #[图片数*预选框总数]

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()  # 计算框位置的iou损失，加和[预选框数]加和
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()  # 计算物体的置信度交叉熵
        loss_cls = (
            self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()  # 计算物体的类别损失交叉熵
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                        cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all] 如果为1说明是通过筛选的预选框，为零则不是
        #   is_in_boxes_and_center  [num_gt, len(fg_mask)]
        # -------------------------------------------------------#
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        # -------------------------------------------------------#
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]    #取出经历过初次筛选后的预选框信息
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]   #得到筛选后的框的数量

        # -------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        # -------------------------------------------------------#
        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)  # 计算真值与预测的iou
        #min、max函数中有广播机制，使得[num_gt, 4]，[fg_mask, 4]在广播下统一尺寸。
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # 转化为损失形式

        # -------------------------------------------------------#
        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        # -------------------------------------------------------#
        cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(
            num_gt, 1, 1).sigmoid_()    #使其对其到真值框的数目，以便对于每一个真值框都遍历比较大小。
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64),
                                     self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        #真值的独热编码
        pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        #计算两个的交叉熵
        del cls_preds_

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()
        #计算损失，如果不同时满足框中心在真值框且框中心与真值的中心距离近，给一个很高的惩罚

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):  # 得到两个框的iou，通过参数确定处理的坐标形式是指明左上右下还是中心+宽高
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])  #max也有广播机制
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
                          center_radius=2.5):  # 用于获取哪些存在于方格内
        #得到属于正样本的特征点会落在物体真实框内部，特征点中心与物体真实框中心相近的预选框

        # 长度为预选框总数目,数值为方格大小。
        expanded_strides_per_image = expanded_strides[0]
        # 格式为，真值框数目，预选框总数目，数值为所落在单元格中心的x坐标
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        # 真值框数目，预选框总数目，数值为所落在单元格中心的y坐标
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        # 格式为，真值框数目，预选框总数目，数值为真值框的上下左右，将其复制预选框数目那么多份
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)

        b_l = x_centers_per_image - gt_bboxes_per_image_l  # 计算各个坐标的差，每一行代表一个真值框，行中每一个元素代表在这个元素上的差距。
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # 将四个参数堆叠，变为[真值框数目，预选框总数目,4]的格式

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # 格式为[真值框数目，预选框数目]可以在其中找到为1的就是真值框中心所落入的预选框。
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  # 选出所有有真值点落入的预选框，格式[预选框数目]

        #   格式为，真值框数目，预选框总数目，数值为真值框中心的一个范围，将其复制预选框数目那么多份
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1) \
                                    .repeat(1,
                                            total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1) \
                                    .repeat(1,
                                            total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1) \
                                    .repeat(1,
                                            total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1) \
                                    .repeat(1,
                                            total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)

        # -------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        c_l = x_centers_per_image - gt_bboxes_per_image_l  # 计算各个坐标与按中心范围的差，每一行代表一个真值框，行中每一个元素代表在这个元素上的差距。
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        # -------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        # -------------------------------------------------------#
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # -------------------------------------------------------#
        #   is_in_boxes_anchor      [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        # -------------------------------------------------------#
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]  #在中心或者中心附近里同时满足两个要求的部分
        return is_in_boxes_anchor, is_in_boxes_and_center   #取出在中心或者中心附近的所有框。

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # -------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        # -------------------------------------------------------#
        matching_matrix = torch.zeros_like(cost)

        # ------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        # ------------------------------------------------------------#
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1) #对于每个真值框，选出最大的10或以下个重合框
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)         #计算对于每个真值框，这些重合框的iou之和

        for gt_idx in range(num_gt):
            # ------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            # ------------------------------------------------------------#
            # 按照每个真值框中最大的k个框iou的和的大小，选出相应数量代价最小的预选框
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # ------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        # ------------------------------------------------------------#
        anchor_matching_gt = matching_matrix.sum(0) #数值表示每个预选框与几个真值框匹配
        if (anchor_matching_gt > 1).sum() > 0:
            # ------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            # ------------------------------------------------------------#
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        # ------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        # ------------------------------------------------------------#
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0  #代表正的预选框的个数
        num_fg = fg_mask_inboxes.sum().item()

        # ------------------------------------------------------------#
        #   对fg_mask进行更新
        # ------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes  #更新成为了所有正样本预选框

        # ------------------------------------------------------------#
        #   获得特征点对应的物品种类
        # ------------------------------------------------------------#
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0) #得到预选框所拟合的真值框的对应[预选框数目]，数值为真值框序号
        gt_matched_classes = gt_classes[matched_gt_inds]    #得到对应的真值框的种类

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]    #得到对应拟合真值框的iou
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def forward(self, inputs, labels=None):  # input尺寸为[batch_size,num_class+5,20/40/80,20/40/80]若干方格下的位置+类别
        outputs = []  # 尺寸为[batch_size,400/1600/6400,num_class+5]若干方格下的位置+类别损失
        x_shifts = []  # 尺寸为[batch_size,400/1600/6400]
        y_shifts = []  # 尺寸为[batch_size,400/1600/6400]
        expanded_strides = []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])  # 记录x的偏移方格数
            y_shifts.append(grid[:, :, 1])  # 记录y的偏移方格数
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)  # [k,w*d]
            outputs.append(output)  # 记录转化为图片根据像素的位置

        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1))


#定义网络参数初始化的类型
def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)