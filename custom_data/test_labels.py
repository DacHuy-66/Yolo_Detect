import cv2
import numpy as np
import os


def draw_bbox(image_path, label_path, output_path):
    """
    Vẽ bounding box lên ảnh và lưu để kiểm tra trực quan

    Args:
        image_path: Đường dẫn tới ảnh
        label_path: Đường dẫn tới file label YOLO
        output_path: Đường dẫn lưu ảnh kết quả
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Đọc file label
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, w, h = map(float, line.strip().split())

            # Chuyển từ tọa độ YOLO (normalized) sang pixel
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            w = int(w * width)
            h = int(h * height)

            # Tính tọa độ góc trên bên trái và góc dưới bên phải
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Vẽ bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Thêm text hiển thị class_id và tọa độ
            text = f"Class: {int(class_id)}, ({x_center / width:.3f}, {y_center / height:.3f}, {w / width:.3f}, {h / height:.3f})"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Lưu ảnh
    cv2.imwrite(output_path, img)


def validate_dataset_bbox(original_dir, resized_dir, output_dir):
    """
    Kiểm tra bounding box trên cả dataset gốc và đã resize
    """
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)

    # Xử lý từng ảnh trong dataset
    images_dir = os.path.join(original_dir, "images")
    labels_dir = os.path.join(original_dir, "labels")

    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Đường dẫn cho dataset gốc
            orig_img_path = os.path.join(images_dir, img_file)
            orig_label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            orig_output_path = os.path.join(output_dir, 'original_' + img_file)

            # Đường dẫn cho dataset đã resize
            resized_img_path = os.path.join(resized_dir, "images", img_file)
            resized_label_path = os.path.join(resized_dir, "labels", os.path.splitext(img_file)[0] + '.txt')
            resized_output_path = os.path.join(output_dir, 'resized_' + img_file)

            # Vẽ bbox cho cả ảnh gốc và ảnh đã resize
            if os.path.exists(orig_label_path):
                draw_bbox(orig_img_path, orig_label_path, orig_output_path)
            if os.path.exists(resized_label_path):
                draw_bbox(resized_img_path, resized_label_path, resized_output_path)


if __name__ == "__main__":
    original_dir = "../data/test"  # Thư mục dataset gốc
    resized_dir = "../data/test"  # Thư mục dataset đã resize
    output_dir = "../data/bbox_validation"  # Thư mục lưu ảnh kiểm tra

    validate_dataset_bbox(original_dir, resized_dir, output_dir)