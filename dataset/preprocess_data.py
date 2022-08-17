import argparse

from dataset.utils import prevent_data_leakage, group_categories
from path_config import PathConfig

pathConfig = PathConfig()


def preprocessing_data(parse) -> None:
    prevent_data_leakage(annotation_path=pathConfig.all_annotation_path, save_path=pathConfig.cleaned_annotation_path)

    if parse.num_classes == 2:
        group_categories(annotation_path=pathConfig.cleaned_annotation_path,
                         save_path=pathConfig.annotation_2_classes_path,
                         num_classes=2)

    if parse.num_classes == 3:
        group_categories(annotation_path=pathConfig.cleaned_annotation_path,
                         save_path=pathConfig.annotation_3_classes_path,
                         num_classes=3)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', required=True, type=int, default=2, help='number of classes: 2 or 3')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    preprocessing_data(opt)
