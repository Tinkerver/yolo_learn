import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction='none', loss_type='iou'):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self,pred,target):
        assert pred.shape[0]==target.shape[0]

        pred=pred.view(-1,4)
        target=target.view(-1,4)
        tl=torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))  #计算重合框左上
        br=torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))  #计算重合框右下

        area_p=torch.prod(pred[:, 2:], 1)   #预测框面积
        area_g=torch.prod(target[:,2:],1)   #真值框面积

        en = (tl < br).type(tl.type()).prod(dim=1)  #返回两框是否相交
        area_i=torch.prod(br-tl,1)*en   # 相交的会返回大小，未相交的返回零
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)   #得到iou

        if self.loss_type=='iou':
            loss=1-iou**2
        elif self.loss_type=='qiou':
            c_tl=torch.min((pred[:,:2]-pred[:,2:]/2),(target[:,:2]-target[:,2:]/2)) #计算大框左上
            c_br=torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))    #计算大框右下
            area_c=torch.prod(c_br-c_tl,1)#大框面积
            giou=iou-(area_c-area_u)/area_c.clamp(1e-16)    #截断下届
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
















