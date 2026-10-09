'''
Author: xiafanli xiafanli@eccom.com.cn
Date: 2024-01-23 16:35:08
Description: 训练脚本
'''
from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones import STDC1
import paddleseg.transforms as T
from paddleseg.datasets import Dataset
from paddleseg.models.losses import CrossEntropyLoss,OhemCrossEntropyLoss
import paddle
from paddleseg.core import train
backbone = STDC1(pretrained="F:\weights\pp_STDCnet1\PP_STDCNet1\model.pdparams")
 
#构建模型
model = PPLiteSeg(num_classes=2,
                 backbone= backbone,
                 arm_out_chs = [32, 64, 128],
                 seg_head_inter_chs = [32, 64, 64],
                 pretrained=None)
 
# 构建训练用的transforms
transforms = [
    T.ResizeStepScaling(min_scale_factor=0.5,max_scale_factor=2.5,scale_step_size=0.25),
    T.RandomPaddingCrop(crop_size=[960,720]),
    T.RandomHorizontalFlip(),
    T.RandomDistort(brightness_range=0.5,contrast_range=0.5,saturation_prob=0.5),
    T.Normalize()
]
 
# 构建训练集
train_dataset = Dataset(
    transforms = transforms,
    dataset_root = 'F:/datasets/rope_ladder',
    num_classes= 2,
    train_path = 'F:/datasets/rope_ladder/train.txt',
    mode='train'
)
 
# 构建验证用的transforms
transforms = [
    T.Normalize()
]
 
# 构建验证集
val_dataset = Dataset(
    transforms = transforms,
    dataset_root = 'F:/datasets/rope_ladder',
    num_classes= 2,
    val_path = 'F:/datasets/rope_ladder/val.txt',
    mode='val'
)
 
# 设置学习率
base_lr = 0.01
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=1000, end_lr=0)
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
 
#构建损失函数
losses = {}
losses['types'] = [OhemCrossEntropyLoss()] * 3
losses['coef'] = [1]* 3
 
#设置训练函数
train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output',
    iters=10000,
    batch_size=4,
    save_interval=200,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)