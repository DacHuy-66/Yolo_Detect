Xây dựng hệ thống theo dõi đối tượng trong video sử dụng YOLO 
1. Giới thiệu đề tài
Đề tài: Xây dựng hệ thống theo dõi đối tượng trong video sử dụng YOLO.
Mục tiêu:

Phát hiện và theo dõi các đối tượng trong video theo thời gian thực.
Áp dụng thuật toán YOLO (You Only Look Once) để nhận dạng và xác định các vật thể trong video.
Kết hợp kỹ thuật theo dõi đối tượng (Object Tracking) để duy trì ID đối tượng qua các khung hình.
Ứng dụng:

Giám sát an ninh (camera giám sát).
Phát hiện và đếm số lượng người hoặc vật thể trong video.
Ứng dụng trong xe tự lái và quản lý giao thông.
2. Công nghệ sử dụng
YOLOv8: Mô hình phát hiện vật thể nhanh và chính xác.
Python: Ngôn ngữ lập trình chính.
OpenCV: Thư viện xử lý video và hình ảnh.
TensorFlow/Keras hoặc PyTorch: Framework hỗ trợ huấn luyện và triển khai mô hình YOLO.
Numpy, Pandas: Xử lý dữ liệu.
Roboflow: Tiền xử lý và chuẩn hóa dữ liệu (nếu cần).
3. Các bước thực hiện
3.1. Chuẩn bị dữ liệu
Thu thập video chứa các đối tượng cần theo dõi.
Chuyển đổi video thành từng khung hình (frames).
Gắn nhãn dữ liệu (Bounding Boxes) bằng công cụ như LabelImg hoặc sử dụng dữ liệu sẵn có (COCO, Pascal VOC).
Định dạng dữ liệu phù hợp với YOLO format.
3.2. Huấn luyện mô hình YOLO
Sử dụng YOLOv8 để huấn luyện mô hình phát hiện đối tượng.
Chia dữ liệu thành 3 phần:
Training set (huấn luyện mô hình).
Validation set (đánh giá mô hình).
Test set (kiểm thử mô hình).
Tùy chỉnh siêu tham số (hyperparameters):
Learning rate, Batch size, Số epoch.
Sử dụng hàm mất mát:
box_loss (Localization Loss).
cls_loss (Classification Loss).
dfl_loss (Distribution Focal Loss).
3.3. Phát hiện và theo dõi đối tượng
Sử dụng YOLO để phát hiện vật thể trong từng khung hình của video.
Tích hợp Object Tracking:
Các thuật toán như Deep SORT, Kalman Filter hoặc ByteTrack để gán ID và theo dõi các đối tượng qua nhiều khung hình.
3.4. Hiển thị kết quả
Vẽ các Bounding Boxes trên video để hiển thị vị trí và ID của đối tượng.
Xuất video kết quả với đối tượng được theo dõi.
