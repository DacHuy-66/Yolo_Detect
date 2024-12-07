import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import threading
import argparse

class ObjectDetectionGUI:
    def __init__(self, root):
        # Khởi tạo cửa sổ chính
        self.root = root
        self.root.title("Object Detection GUI")
        
        # Khởi tạo các biến cần thiết
        self.video_path = None        # Đường dẫn đến file video
        self.is_playing = False       # Trạng thái phát video
        self.cap = None               # Đối tượng để đọc video
        self.selected_class = tk.StringVar()  # Biến lưu class được chọn
        
        # Tạo giao diện
        self.create_widgets()
        
        # Cấu hình các tham số cho model
        self.args = argparse.Namespace(
            weights='runs/train/vehicle_person/weights/best.pt',  # Đường dẫn đến file weights
            data='data.yaml',                                     # File cấu hình data
            conf_thres=0.5,                                      # Ngưỡng tin cậy
            device=''                                            # Thiết bị xử lý (GPU/CPU)
        )
        
        # Khởi tạo model và detector
        self.setup_detector()

    def create_widgets(self):
        # Tạo frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Nút chọn video
        self.select_btn = ttk.Button(main_frame, text="Chọn Video", command=self.select_video)
        self.select_btn.grid(row=0, column=0, pady=5)
        
        # Label hiển thị đường dẫn video
        self.path_label = ttk.Label(main_frame, text="Chưa chọn video")
        self.path_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Nút Start/Stop
        self.start_btn = ttk.Button(main_frame, text="Start", command=self.toggle_detection, state=tk.DISABLED)
        self.start_btn.grid(row=0, column=2, pady=5)
        
        # Frame chứa phần chọn class
        class_frame = ttk.Frame(main_frame)
        class_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Label cho phần chọn class
        ttk.Label(class_frame, text="Chọn đối tượng tracking:").grid(row=0, column=0, padx=5)
        
        # Listbox để chọn nhiều class
        self.class_listbox = tk.Listbox(class_frame, selectmode=tk.MULTIPLE, height=5)
        self.class_listbox.grid(row=0, column=1, padx=5)
        
        # Frame hiển thị video
        self.video_frame = ttk.Label(main_frame)
        self.video_frame.grid(row=1, column=0, columnspan=3, pady=10)

    def setup_detector(self):
        # Khởi tạo model với GPU nếu có
        self.model = DetectMultiBackend(
            weights=self.args.weights,
            device=torch.device(self.args.device) if self.args.device else \
                   torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            fuse=True
        )
        self.model = AutoShape(self.model)  # Tự động xử lý kích thước ảnh đầu vào
        self.tracker = DeepSort(max_age=30)  # Khởi tạo tracker với max_age=30 frames
        
        # Đọc danh sách classes từ file
        try:
            with open('data_ext/classes.names', 'r') as f:
                self.class_names = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            print("Không tìm thấy file classes.names")
            self.class_names = []
        
        # Tạo màu ngẫu nhiên cho mỗi class
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))
        
        # Xóa danh sách cũ trong listbox
        self.class_listbox.delete(0, tk.END)
        
        # Chỉ thêm person và vehicle vào listbox
        for class_name in self.class_names:
            if class_name.lower() in ['person', 'vehicle']:
                self.class_listbox.insert(tk.END, class_name)
        
        # Chọn class đầu tiên làm mặc định
        if self.class_names:
            self.class_listbox.selection_set(0)

    def select_video(self):
        # Mở hộp thoại chọn file video
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            # Cập nhật UI khi đã chọn video
            self.path_label.config(text=self.video_path)
            self.start_btn.config(state=tk.NORMAL)
            # Đóng video cũ nếu đang mở
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_playing = False
            self.start_btn.config(text="Start")

    def toggle_detection(self):
        # Xử lý khi nhấn nút Start/Stop
        if not self.is_playing:
            if self.video_path:
                self.is_playing = True
                self.start_btn.config(text="Stop")
                # Chạy detection trong thread riêng
                threading.Thread(target=self.run_detection, daemon=True).start()
        else:
            # Dừng detection
            self.is_playing = False
            self.start_btn.config(text="Start")
            if self.cap:
                self.cap.release()
                self.cap = None

    def get_selected_classes(self):
        # Lấy danh sách các class được chọn từ listbox
        selected_indices = self.class_listbox.curselection()
        return [self.class_names[i] for i in selected_indices]

    def run_detection(self):
        # Khởi tạo đối tượng đọc video
        self.cap = cv2.VideoCapture(self.video_path)
        
        while self.is_playing:
            # Đọc từng frame
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Lấy danh sách class được chọn
            selected_classes = self.get_selected_classes()
            
            # Xử lý frame
            processed_frame = self.process_frame(
                frame, self.model, self.tracker,
                self.class_names, self.colors, self.args.conf_thres,
                selected_classes=selected_classes
            )
            
            # Chuyển đổi frame để hiển thị trong GUI
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Điều chỉnh kích thước frame cho phù hợp với cửa sổ
            window_width = self.root.winfo_width() - 40
            window_height = self.root.winfo_height() - 100
            frame_pil.thumbnail((window_width, window_height))
            
            # Hiển thị frame
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.video_frame.configure(image=frame_tk)
            self.video_frame.image = frame_tk
            
            # Cập nhật GUI
            self.root.update()
            
        # Giải phóng video khi kết thúc
        if self.cap:
            self.cap.release()

    def process_frame(self, frame, model, tracker, class_names, colors, conf_thres, selected_classes=None):
        # Thực hiện detection trên frame
        results = model(frame)
        det = results.pred[0]
        
        if len(det):
            detections = []
            for *xyxy, confidence, class_id in det:
                try:
                    class_id = int(class_id)
                    # Chỉ xử lý class_id 2 và 3 (person và vehicle)
                    if class_id not in [2, 3]:
                        continue
                    
                    # Lấy tên class và kiểm tra có được chọn không
                    class_name = class_names[class_id - 2]
                    if selected_classes and class_name.lower() not in [cls.lower() for cls in selected_classes]:
                        continue
                    if confidence < conf_thres:
                        continue
                    
                    # Chuyển đổi tọa độ bounding box
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append(([x1, y1, x2-x1, y2-y1], confidence, class_id))
                except Exception as e:
                    print(f"Lỗi xử lý detection: {e}")
                    continue
            
            # Cập nhật tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Vẽ các tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                try:
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    
                    # Vẽ bounding box
                    x1, y1, x2, y2 = map(int, ltrb)
                    det_class = int(track.get_det_class()) - 2
                    
                    if det_class < 0 or det_class >= len(colors):
                        continue
                    
                    color = colors[det_class]
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Vẽ nhãn
                    text = f'{class_names[det_class]}-{track_id}'
                    # Vẽ nhãn với độ chính xác
                    confidence = track.get_det_conf()
                    text = f'{class_names[det_class]}-{track_id}({confidence:.2f})'
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                except Exception as e:
                    print(f"Lỗi xử lý track: {e}")
                    continue
                
        return frame

def main():
    # Khởi tạo cửa sổ chính
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    # Chạy ứng dụng
    root.mainloop()

if __name__ == "__main__":
    main() 