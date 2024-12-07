import os
from PIL import Image
import shutil


def adjust_bbox_coordinates(original_size, new_size, coordinates, padding, resize_factor):
    """
    Adjust bounding box coordinates when resizing image.
    YOLO format: class_id, x_center, y_center, width, height (normalized 0-1)
    """
    orig_w, orig_h = original_size
    new_w, new_h = new_size

    class_id = int(coordinates[0])
    x_center, y_center, width, height = coordinates[1:5]  # Normalized

    # Denormalize coordinates to original image size
    x_center *= orig_w
    y_center *= orig_h
    width *= orig_w
    height *= orig_h

    # Apply resizing factor
    x_center = (x_center * resize_factor) + padding[0]
    y_center = (y_center * resize_factor) + padding[1]
    width *= resize_factor
    height *= resize_factor

    # Normalize coordinates to new image size
    x_center /= new_w
    y_center /= new_h
    width /= new_w
    height /= new_h

    # Ensure values stay within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return [class_id, x_center, y_center, width, height]


def process_label_file(input_path, output_path, original_size, new_size, padding, resize_factor):
    """
    Process and adjust coordinates in label file
    """
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            values = [float(x) for x in line.strip().split()]
            new_values = adjust_bbox_coordinates(original_size, new_size, values, padding, resize_factor)
            f_out.write(' '.join(f"{x:.6f}" for x in new_values) + '\n')


def resize_dataset(input_root, output_root, target_size=(416, 416)):
    """
    Resize dataset and adjust coordinates in labels while maintaining aspect ratio
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    images_dir = os.path.join(output_root, "images")
    labels_dir = os.path.join(output_root, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    input_images_dir = os.path.join(input_root, "images")
    input_labels_dir = os.path.join(input_root, "labels")

    for img_file in os.listdir(input_images_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_img_path = os.path.join(input_images_dir, img_file)
            output_img_path = os.path.join(images_dir, img_file)

            try:
                with Image.open(input_img_path) as img:
                    original_size = img.size
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Maintain aspect ratio while resizing
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    resize_factor = min(target_size[0] / original_size[0], target_size[1] / original_size[1])

                    # Create new image with target size and paste resized image in center
                    new_img = Image.new('RGB', target_size, (0, 0, 0))
                    paste_x = (target_size[0] - img.size[0]) // 2
                    paste_y = (target_size[1] - img.size[1]) // 2
                    new_img.paste(img, (paste_x, paste_y))
                    new_img.save(output_img_path, quality=95)

                    # Process corresponding label file
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    input_label_path = os.path.join(input_labels_dir, label_file)
                    output_label_path = os.path.join(labels_dir, label_file)

                    if os.path.exists(input_label_path):
                        process_label_file(
                            input_label_path,
                            output_label_path,
                            original_size,
                            target_size,
                            (paste_x, paste_y),
                            resize_factor,
                        )

            except Exception as e:
                print(f"Error processing {input_img_path}: {str(e)}")


if __name__ == "__main__":
    input_root = "dataset/valid"
    output_root = "data/valid"
    target_size = (320, 320)

    resize_dataset(input_root, output_root, target_size)
