
exp_configs = {
    "Faster_RCNN": {
        "name": "Faster_RCNN",
        "wandb_init": {
            "project": "Cancer_Detection",
            "entity": "thesisltran",
            "name": "Faster_RCNN",
            "id": "Faster_RCNN",
            "notes": "faster_rcnn_x101_64x4d_fpn_1x_coco",
            "save_code": True
        },
        "config_path": "configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py",
        "real_checkpoint_path": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth",
        "checkpoint_path": "/content/mmdetection/checkpoints/faster_rcnn_x101_64x4d_fpn_1x_coco.pth"
    },
    "RetinaNet": {
        "name": "RetinaNet",
        "wandb_init": {
            "project": "Cancer_Detection",
            "entity": "thesisltran",
            "name": "RetinaNet",
            "id": "RetinaNet",
            "notes": "retinanet_x101_64x4d_fpn_1x_coco",
            "save_code": True
        },
        "config_path": "configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py",
        "real_checkpoint_path": "https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth",
        "checkpoint_path": "/content/mmdetection/checkpoints/retinanet_x101_64x4d_fpn_1x_coco.pth"
    },
    "VarifocalNet": {
        "name": "VarifocalNet",
        "wandb_init": {
            "project": "Cancer_Detection",
            "entity": "thesisltran",
            "name": "VarifocalNet",
            "id": "VarifocalNet",
            "notes": "vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco",
            "save_code": True
        },
        "config_path": "configs/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py",
        "real_checkpoint_path": "https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth",
        "checkpoint_path": "/content/mmdetection/checkpoints/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.pth"
    },
    "RetinaNet_Swin": {
        "name": "RetinaNet_Swin",
        "wandb_init": {
            "project": "Cancer_Detection",
            "entity": "thesisltran",
            "name": "RetinaNet_Swin",
            "id": "RetinaNet_Swin",
            "notes": "retinanet_x101_64x4d_fpn_1x_coco",
            "save_code": True
        },
        "config_path": "configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py",
        "real_checkpoint_path": "",
        "checkpoint_path": ""
    }
    ,
    "RetinaNet_Swin_T": {
        "name": "RetinaNet_Swin_T",
        "wandb_init": {
            "project": "Cancer_Detection",
            "entity": "thesisltran",
            "name": "RetinaNet_Swin_T",
            "id": "RetinaNet_Swin_T",
            "notes": "retinanet_x101_64x4d_fpn_1x_coco",
            "save_code": True
        },
        "config_path": "configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py",
        "real_checkpoint_path": "",
        "checkpoint_path": ""
    }
    ,
    "Deformable_DETR": {
        "name": "Deformable_DETR",
        "wandb_init": {
            "project": "Cancer_Detection",
            "entity": "thesisltran",
            "name": "Deformable_DETR",
            "id": "Deformable_DETR",
            "notes": "deformable_detr_r50_16x2_50e_coco",
            "save_code": True
        },
        "config_path": "configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py",
        "real_checkpoint_path": "https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth",
        "checkpoint_path": "/content/mmdetection/checkpoints/deformable_detr_r50_16x2_50e_coco.pth"
    }
}
