import argparse

from dataset.utils import split_data
from path_config import PathConfig

pathConfig = PathConfig()


def split_train_test_val(parse):
    num_classes = parse.num_classes

    if num_classes == 2:
        split_data(
            annotation_path=pathConfig.annotation_2_classes_path,
            train_annotation_path=pathConfig.train_annotation_2_classes_path,
            test_annotation_path=pathConfig.test_annotation_2_classes_path,
            val_annotation_path=pathConfig.val_annotation_2_classes_path
        )

    if num_classes == 3:
        split_data(
            annotation_path=pathConfig.annotation_3_classes_path,
            train_annotation_path=pathConfig.train_annotation_3_classes_path,
            test_annotation_path=pathConfig.test_annotation_3_classes_path,
            val_annotation_path=pathConfig.val_annotation_3_classes_path
        )


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', required=True, type=int, default=2, help='number of classes: 2 or 3')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    split_train_test_val(opt)
