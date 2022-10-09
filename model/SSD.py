from typing import Dict

from mmdet.apis import set_random_seed
from pathlib import Path
from mmcv import Config


def get_ssd_config(
        data_config: Dict,
        num_classes: int,
        img_size: int,
        max_epochs: int = 12,
        lr: float = 0.0025,
        pretrained: bool = True,
):
    if num_classes == 2:
        classes = ['normal', 'cancer']
    if num_classes == 3:
        classes = ['normal', 'cancer', 'suspected_cancer']

    cfg = Config.fromfile('/content/mmdetection/configs/ssd/ssd300_coco.py')

    cfg.dataset_type = 'CocoDataset'
    cfg.classes = classes
    cfg.data_root = data_config['data_root']

    # modify num classes of the model in box head
    cfg.model.bbox_head.num_classes = num_classes

    cfg.data.train.ann_file = data_config['train_annotation_file']
    cfg.data.train.img_prefix = data_config['train_image_path']
    cfg.data.train.classes = classes
    cfg.data.train.type = 'CocoDataset'

    cfg.data.val.ann_file = data_config['val_annotation_file']
    cfg.data.val.img_prefix = data_config['val_image_path']
    cfg.data.val.classes = classes
    cfg.data.val.type = 'CocoDataset'

    cfg.data.test.ann_file = data_config['val_annotation_file']
    cfg.data.test.img_prefix = data_config['val_image_path']
    cfg.data.test.classes = classes
    cfg.data.test.type = 'CocoDataset'

    cfg.data.samples_per_gpu = 2
    cfg.data.workers_per_gpu = 2

    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    if pretrained:
        cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth'
    else:
        cfg.load_from = ''

    cfg.work_dir = './tutorial_exps'

    cfg.optimizer.lr = cfg.optimizer.lr / 8
    cfg.lr_config.warmup = "linear"
    cfg.lr_config.warmup_iters = 1000
    cfg.lr_config.warmup_ratio = 0.001
    cfg.log_config.interval = 200

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'bbox'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 1

    cfg.runner.max_epochs = max_epochs

    cfg.seed = 0
    set_random_seed(0, deterministic=True)
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'

    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.val.pipeline = cfg.test_pipeline

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={'project': 'Cancer_Detection',
                          'name': 'SSD_' + str(num_classes) + "_" + str(img_size) + "_" + str(pretrained),
                          'id': 'SSD_' + str(num_classes) + "_" + str(img_size) + "_" + str(pretrained),
                          'save_code': True,
                          'tags': [str(num_classes), str(img_size), "SSD", str(pretrained)]
                          },
             interval=10,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=50)]

    return cfg