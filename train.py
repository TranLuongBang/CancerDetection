from mmcv import Config
import mmcv
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp

def run():
    cfg = Config.fromfile('./CancerDetection/model/3_classes/faster_rcnn_x101_64x4d_fpn_1x_coco.py')

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES


    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, 'faster_rcnn_x101_64x4d_fpn_1x_coco.py'))

    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':
    run()