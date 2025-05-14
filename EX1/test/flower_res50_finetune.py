# 引入基础配置
_base_ = [
    "mmpretrain/configs/_base_/models/resnet50.py",
    "mmpretrain/configs/_base_/datasets/imagenet_bs32.py",
    "mmpretrain/configs/_base_/schedules/imagenet_bs256.py",
    "mmpretrain/configs/_base_/default_runtime.py"
]


# 1. 模型输出设置
model = dict(
    head=dict(num_classes=5, topk=(1,),loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1))
)

load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

# 2. 图像预处理配置
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    num_classes=5
)

# 3. 数据加载配置
dataset_type = 'CustomDataset'
data_root = 'flower_dataset'
classes = [c.strip() for c in open(f'{data_root}/classes.txt')]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='flower_dataset',  # 添加data_root
        data_prefix='',  # 修改为相对路径
        ann_file='train.txt',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs'),
        ]
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='flower_dataset',  # 添加data_root
        data_prefix='',  # 修改为相对路径
        ann_file='val.txt',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]
    )
)
val_cfg = dict()
val_evaluator = dict(type='Accuracy', topk=(1,))

# 4. 优化器配置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4)
)

# 5. 学习率自动缩放（可选）
auto_scale_lr = dict(base_batch_size=256)

# 6. 训练过程配置
train_cfg = dict(
    by_epoch=True,
    max_epochs=15,
    val_interval=1,
)

param_scheduler = dict( 
     type='MultiStepLR',
     by_epoch=True,
     milestones=[5, 10],
     gamma=0.1
)
