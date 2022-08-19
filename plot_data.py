import argparse

import fiftyone as fo

from path_config import PathConfig

pathConfig = PathConfig()


def plot_data(parse) -> None:
    img_path = parse.img_path
    annotation_path = parse.annotation_path
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=img_path,
        labels_path=annotation_path,
        include_id=True,
        label_field="detections",
    )

    session = fo.launch_app(coco_dataset)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True, type=str, help='The location of the images for certain dataset')
    parser.add_argument('--annotation_path', required=True, type=str,
                        help='The location of the annotation for certain dataset')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    plot_data(opt)
