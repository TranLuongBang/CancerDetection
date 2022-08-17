from pathlib import Path


class PathConfig:

    def __init__(self):
        self.data_path = Path('./content/driver/MyDrive')

        self.image_folder_path = self.data_path / "dataset" / "images"
        self.annotation_folder_path = self.data_path / "dataset" / "annotation"

        self.all_images_path = self.image_folder_path / "all_images"
        self.all_annotation_path = self.annotation_folder_path / "annotations.json"

        self.cleaned_annotation_path = self.annotation_folder_path / "cleaned_annotation_s.json"
        self.annotation_2_classes_path = self.annotation_folder_path / "annotations_2_classes.json"
        self.annotation_3_classes_path = self.annotation_folder_path / "annotations_3_classes.json"

        self.train_image_path = self.image_folder_path / "train_images"
        self.train_annotation_2_classes_path = self.annotation_folder_path / "train_annotation_2_classes.json"
        self.train_annotation_3_classes_path = self.annotation_folder_path / "train_annotation_3_classes.json"

        self.test_image_path = self.image_folder_path / "test_images"
        self.test_annotation_2_classes_path = self.annotation_folder_path / "test_annotation_2_classes.json"
        self.test_annotation_3_classes_path = self.annotation_folder_path / "test_annotation_3_classes.json"

        self.val_image_path = self.image_folder_path / "val_images"
        self.val_annotation_2_classes_path = self.annotation_folder_path / "val_annotation_2_classes.json"
        self.val_annotation_3_classes_path = self.annotation_folder_path / "val_annotation_3_classes.json"

        self.data_2_classes = self.data_path / "data_2_classes"

        self.data_3_classes = self.data_path / "data_3_classes"

        self.size_256_image_path = Path("size_256/images")
        self.size_256_annotation_path = Path("size_256/annotations")
        self.size_256_train_image_path = self.size_256_image_path / "train_images"
        self.size_256_test_image_path = self.size_256_image_path / "test_images"
        self.size_256_val_image_path = self.size_256_image_path / "val_images"
        self.size_256_train_annotation_path = self.size_256_annotation_path / "train_annotations.json"
        self.size_256_test_annotation_path = self.size_256_annotation_path / "test_annotations.json"
        self.size_256_val_annotation_path = self.size_256_annotation_path / "val_annotations.json"

        self.size_512_image_path = Path('size_512/images')
        self.size_512_annotation_path = Path('size_512/annotations')
        self.size_512_train_image_path = self.size_512_image_path / "train_images"
        self.size_512_test_image_path = self.size_512_image_path / "test_images"
        self.size_512_val_image_path = self.size_512_image_path / "val_images"
        self.size_512_train_annotation_path = self.size_512_annotation_path / "train_annotations.json"
        self.size_512_test_annotation_path = self.size_512_annotation_path / "test_annotations.json"
        self.size_512_val_annotation_path = self.size_512_annotation_path / "val_annotations.json"

        self.size_640_image_path = Path('size_640/images')
        self.size_640_annotation_path = Path('size_640/annotations')
        self.size_640_train_image_path = self.size_640_image_path / "train_images"
        self.size_640_test_image_path = self.size_640_image_path / "test_images"
        self.size_640_val_image_path = self.size_640_image_path / "val_images"
        self.size_640_train_annotation_path = self.size_640_annotation_path / "train_annotations.json"
        self.size_640_test_annotation_path = self.size_640_annotation_path / "test_annotations.json"
        self.size_640_val_annotation_path = self.size_640_annotation_path / "val_annotations.json"

        self.size_1024_image_path = Path('size_1024/images')
        self.size_1024_annotation_path = Path('size_1024/annotations')
        self.size_1024_train_image_path = self.size_1024_image_path / "train_images"
        self.size_1024_test_image_path = self.size_1024_image_path / "test_images"
        self.size_1024_val_image_path = self.size_1024_image_path / "val_images"
        self.size_1024_train_annotation_path = self.size_1024_annotation_path / "train_annotations.json"
        self.size_1024_test_annotation_path = self.size_1024_annotation_path / "test_annotations.json"
        self.size_1024_val_annotation_path = self.size_1024_annotation_path / "val_annotations.json"
