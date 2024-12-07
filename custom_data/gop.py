import os
import shutil
from pathlib import Path


def modify_car_labels(input_dir, output_dir, new_class_id=2):
    """
    Modify car detection labels by changing class ID from 0 to new_class_id
    """
    os.makedirs(output_dir, exist_ok=True)

    # Process each label file
    for label_file in os.listdir(input_dir):
        if not label_file.endswith('.txt'):
            continue

        input_path = os.path.join(input_dir, label_file)
        output_path = os.path.join(output_dir, label_file)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        # Modify class ID in each line
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure valid YOLO format
                parts[0] = str(new_class_id)  # Change class ID
                modified_lines.append(' '.join(parts) + '\n')

        # Write modified content
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)


def merge_datasets(car_dataset_path, person_dataset_path, output_path):
    """
    Merge car and person detection datasets
    """
    # Create output directory structure
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

    # Process each split
    for split in splits:
        # Temporary directory for modified car labels
        temp_car_labels = os.path.join(output_path, 'temp_car_labels')
        os.makedirs(temp_car_labels, exist_ok=True)

        # Modify car labels
        car_labels_path = os.path.join(car_dataset_path, split, 'labels')
        modify_car_labels(car_labels_path, temp_car_labels)

        # Copy and merge images and labels
        for dataset_path, is_car in [(car_dataset_path, True), (person_dataset_path, False)]:
            # Copy images
            src_images = os.path.join(dataset_path, split, 'images')
            dst_images = os.path.join(output_path, split, 'images')

            for img in os.listdir(src_images):
                shutil.copy2(
                    os.path.join(src_images, img),
                    os.path.join(dst_images, img)
                )

            # Copy labels
            src_labels = temp_car_labels if is_car else os.path.join(dataset_path, split, 'labels')
            dst_labels = os.path.join(output_path, split, 'labels')

            for label in os.listdir(src_labels):
                shutil.copy2(
                    os.path.join(src_labels, label),
                    os.path.join(dst_labels, label)
                )

        # Clean up temporary directory
        shutil.rmtree(temp_car_labels)

    # Create new data.yaml
    yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images
nc: 3
names: ['pedestrian', 'people', 'car']
"""

    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content)


# Example usage
if __name__ == "__main__":
    car_dataset = "data_car"
    person_dataset = "data_person"
    output_path = "data/dataset"

    merge_datasets(car_dataset, person_dataset, output_path)