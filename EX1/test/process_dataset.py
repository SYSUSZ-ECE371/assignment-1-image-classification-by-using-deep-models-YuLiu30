
import os
import shutil
import random

# Path to the downloaded flower_dataset folder containing the class subfolders
original_dataset_path = 'my_dataset'

# Path where the processed dataset will be created 
output_dataset_path = 'flower_dataset' 

# Ratio for splitting data (training vs validation)
split_ratio = 0.8 # 80% for training, 20% for validation


def process_flower_dataset(original_path, output_path, split_ratio):
    """
    Processes the original flower dataset into ImageNet format.

    Args:
        original_path (str): Path to the original dataset folder.
        output_path (str): Path to create the processed dataset structure.
        split_ratio (float): Ratio of data for the training set (0.0 to 1.0).
    """
    # Ensure output directory is clean or create it
    if os.path.exists(output_path):
        print(f"Warning: Output directory '{output_path}' already exists. Removing it.")
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    os.makedirs(train_path)
    os.makedirs(val_path)

    classes_file = os.path.join(output_path, 'classes.txt')
    train_annotation_file = os.path.join(output_path, 'train.txt')
    val_annotation_file = os.path.join(output_path, 'val.txt')

    # Get class names from folder names
    class_names = sorted([d for d in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes: {class_names}")

    # Write classes.txt
    with open(classes_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Created '{classes_file}'")

    # Prepare annotation lists
    train_annotations = []
    val_annotations = []

    # Process each class
    for class_name in class_names:
        class_original_path = os.path.join(original_path, class_name)
        class_train_path = os.path.join(train_path, class_name)
        class_val_path = os.path.join(val_path, class_name)

        os.makedirs(class_train_path, exist_ok=True)
        os.makedirs(class_val_path, exist_ok=True)

        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_original_path) if os.path.isfile(os.path.join(class_original_path, f))]

        # Shuffle files and split
        random.shuffle(image_files)
        split_index = int(len(image_files) * split_ratio)
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        print(f"Processing class '{class_name}': {len(image_files)} images found, {len(train_files)} for training, {len(val_files)} for validation.")

        # Copy files and generate annotations
        class_idx = class_to_idx[class_name]

        for filename in train_files:
            src_path = os.path.join(class_original_path, filename)
            dest_path = os.path.join(class_train_path, filename)
            shutil.copy(src_path, dest_path)
            relative_path = f'train/{class_name}/{filename}'.replace('\\', '/')  # 强制使用正斜杠
            train_annotations.append(f"{relative_path} {class_idx}\n")

        # Validation files
        for filename in val_files:
            src_path = os.path.join(class_original_path, filename)
            dest_path = os.path.join(class_val_path, filename)
            shutil.copy(src_path, dest_path)
            relative_path = f'val/{class_name}/{filename}'.replace('\\', '/')  # 强制使用正斜杠
            val_annotations.append(f"{relative_path} {class_idx}\n")

    # Write annotation files
    with open(train_annotation_file, 'w') as f:
        f.writelines(train_annotations)
    print(f"Created '{train_annotation_file}' with {len(train_annotations)} entries.")

    with open(val_annotation_file, 'w') as f:
        f.writelines(val_annotations)
    print(f"Created '{val_annotation_file}' with {len(val_annotations)} entries.")

    print("\nDataset processing complete!")
    print(f"Processed dataset is located at '{output_path}'")


if __name__ == "__main__":
    # 获取脚本所在目录的父目录（即 task1）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # 上一级目录（task1）

    # 原始数据集路径（指向 my_dataset）
    original_dataset_path = os.path.join(parent_dir, 'my_dataset')
    # 输出路径（指向 flower_dataset）
    output_dataset_path = os.path.join(parent_dir, 'flower_dataset')

    # 检查路径是否存在
    if not os.path.exists(original_dataset_path):
        print(f"Error: Original dataset path not found at '{original_dataset_path}'")
        print("请确保 my_dataset 文件夹位于 task1 目录下！")
    else:
        process_flower_dataset(original_dataset_path, output_dataset_path, split_ratio)