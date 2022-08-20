import copy
import json
import shutil
from pathlib import Path
from typing import Dict

import sklearn

from path_config import PathConfig

pathConfig = PathConfig()


def read_data(data_path: Path) -> Dict:
    with open(data_path) as json_file:
        data = json.load(json_file)
        return data


def write_data(data: Dict, save_path: Path) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(save_path), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def group_categories(annotation_path: Path, save_path: Path, num_classes: int) -> Dict:
    dataset = read_data(data_path=annotation_path)

    normal_category_ids = [1, 2, 3]
    normal_id = 1

    if num_classes == 2:
        cancer_category_ids = [4, 5, 6]
        cancer_category_id = 2

        for annotation in dataset['annotations']:
            if annotation['category_id'] in normal_category_ids:
                annotation['category_id'] = normal_id
            elif annotation['category_id'] in cancer_category_ids:
                annotation['category_id'] = cancer_category_id

        categories = [
            {
                "id": 1,
                "name": "normal",
                "supercategory": ""
            },
            {
                "id": 2,
                "name": "cancer",
                "supercategory": ""
            }
        ]

        dataset['categories'] = categories

    if num_classes == 3:
        cancer_category_ids = [6]
        cancer_category_id = 2

        suspected_cancer_category_ids = [4, 5]
        suspected_cancer_category_id = 3

        for annotation in dataset['annotations']:
            if annotation['category_id'] in normal_category_ids:
                annotation['category_id'] = normal_id
            elif annotation['category_id'] in cancer_category_ids:
                annotation['category_id'] = cancer_category_id
            elif annotation['category_id'] in suspected_cancer_category_ids:
                annotation['category_id'] = suspected_cancer_category_id

        categories = [
            {
                "id": 1,
                "name": "normal",
                "supercategory": ""
            },
            {
                "id": 2,
                "name": "cancer",
                "supercategory": ""
            },
            {
                "id": 3,
                "name": "suspected_cancer",
                "supercategory": ""
            },
        ]

        dataset['categories'] = categories

    write_data(data=dataset, save_path=save_path)

    return dataset


def prevent_data_leakage(annotation_path: Path, save_path: Path) -> None:
    data = read_data(data_path=annotation_path)

    # duplicate image
    duplicate_images = {}
    for item in data['images']:
        if item['file_name'] not in duplicate_images.keys():
            duplicate_images[item['file_name']] = [item['id']]
        else:
            duplicate_images[item['file_name']].append(item['id'])

    # collect unique image
    unique_images = [image_id[-1] for image_id in duplicate_images.values()]
    unique_images = [image for image in unique_images if image not in [53, 86, 82, 83, 50]] + [38, 39, 34, 20, 24]

    dataset = copy.deepcopy(data)

    dataset['annotations'] = [annotation for annotation in dataset['annotations'] if
                              annotation['image_id'] in unique_images]

    dataset['images'] = [image for image in dataset['images'] if image['id'] in unique_images]

    write_data(data=dataset, save_path=save_path)


def split_data(
        annotation_path: Path,
        train_annotation_path: Path,
        val_annotation_path: Path,
        test_annotation_path: Path
) -> None:
    dataset = read_data(data_path=annotation_path)

    image_ids = [image['id'] for image in dataset['images']]

    image_ids = sklearn.utils.shuffle(image_ids, random_state=42)

    train_annotations = {}
    val_annotations = {}
    test_annotations = {}

    for key, value in dataset.items():

        if key == 'images':
            train_annotations[key] = []
            val_annotations[key] = []
            test_annotations[key] = []
            for image in value:
                image_path = pathConfig.all_images_path / image['file_name']
                if image['id'] in image_ids[:65] and Path(image_path).exists():
                    train_annotations[key].append(image)
                    shutil.copy(image_path, pathConfig.train_image_path / image['file_name'])
                if image['id'] in image_ids[65:77] and Path(image_path).exists():
                    val_annotations[key].append(image)
                    shutil.copy(image_path, pathConfig.val_image_path / image['file_name'])
                if image['id'] in image_ids[77:] and Path(image_path).exists():
                    test_annotations[key].append(image)
                    shutil.copy(image_path, pathConfig.test_image_path / image['file_name'])

        elif key == 'annotations':
            train_annotations[key] = []
            val_annotations[key] = []
            test_annotations[key] = []
            for annotation in value:
                if annotation['image_id'] in image_ids[:65]:
                    train_annotations[key].append(annotation)
                if annotation['image_id'] in image_ids[65:77]:
                    val_annotations[key].append(annotation)
                if annotation['image_id'] in image_ids[77:]:
                    test_annotations[key].append(annotation)

        else:
            train_annotations[key] = value
            val_annotations[key] = value
            test_annotations[key] = value

    write_data(data=train_annotations, save_path=train_annotation_path)
    write_data(data=val_annotations, save_path=val_annotation_path)
    write_data(data=test_annotations, save_path=test_annotation_path)
