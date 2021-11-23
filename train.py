import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    classes_path = 'model_data/voc_classes.txt'
    model_path = 'model_data/yolox_s.pth'
    input_shape = [640, 640]
    phi = 's'
    #   mosaic 马赛克数据增强
    #   YOLOX作者强调要在训练结束前的N个epoch关掉Mosaic。因为Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #   并且Mosaic大量的crop操作会带来很多不准确的标注框，本代码自动会在前90%个epoch使用mosaic，后面不使用。
    #   马赛克数据增强将4张训练图像按一定比例组合成1张。Mosaic是一个新的数据增强技术。这使得模型能够学习如何识别比正常尺寸小的物体。
    mosaic = False
    #   余弦退火学习率,按一个公式确定学习率的衰减
    Cosine_scheduler = False
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段
    #   冻结阶段模型的主干backbone被冻结了，特征提取网络不发生改变，训练neck和head
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    Freeze_lr = 1e-3
    #   解冻阶段所有的参数一起训练
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr = 1e-4
    Freeze_Train = True
    #   线程设置
    num_workers = 4
    #   得到图片路径和标签
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'
    #   得到类的名称和长度
    class_names, num_classes = get_classes(classes_path)
    #   模型设计需要种类数来确定cls尺寸
    model = YoloBody(num_classes, phi)
    #   初始化网络参数
    weights_init(model)

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #验证模型格式一致
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()

    #是否启用GPU
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss = YOLOLoss(num_classes)
    loss_history = LossHistory("logs/")

