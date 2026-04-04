import sys, cv2, time, threading, os, pygame, json, numpy as np
from ultralytics import YOLO
from gtts import gTTS
from PIL import ImageFont, ImageDraw, Image
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

class AIVisionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Hỗ Trợ Nhận Diện Vật Thể")
        self.setStyleSheet("background-color: #1a1a1a; color: #f0f0f0; font-family: Arial;")
        self.resize(1280, 720)
        
        self.trans = {}
        if os.path.exists("translations.json"):
            with open("translations.json", "r", encoding="utf-8") as f:
                self.trans = json.load(f)
        
        try: self.p_font = ImageFont.truetype("arial.ttf", 22)
        except: self.p_font = ImageFont.load_default()

        pygame.mixer.init()
        self.is_speaking = False
        self.is_running = False 
        self.prev_time = 0
        self.init_ui()
        
        self.scan_supported_resolutions()

    def get_cams(self):
        arr = []
        for i in range(2):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened(): arr.append(i); cap.release()
        return arr

    def init_ui(self):
        main_layout = QHBoxLayout()
        
        side_bar = QWidget()
        side_bar.setFixedWidth(320)
        side_bar.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        left_panel = QVBoxLayout(side_bar)
        left_panel.setSpacing(12)
        
        title = QLabel("CÀI ĐẶT ỨNG DỤNG")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #4da6ff; padding-bottom: 10px;")
        left_panel.addWidget(title)

        # Hardware
        self.hw_widget = QWidget()
        hw_layout = QVBoxLayout(self.hw_widget)
        hw_layout.setContentsMargins(0,0,0,0)

        self.cb_cam = QComboBox()
        self.cb_cam.setToolTip("Chọn Camera đang cắm vào máy.")
        [self.cb_cam.addItem(f"Camera {c}", c) for c in self.get_cams()]
        self.cb_cam.currentIndexChanged.connect(self.scan_supported_resolutions)
        
        self.cb_model = QComboBox(); self.cb_model.addItems(["yolov8n.pt (Chính xác thấp)", "yolov8s.pt (Chính xác cao)"])
        
        hw_layout.addWidget(QLabel("Camera đầu vào: ")); hw_layout.addWidget(self.cb_cam)
        hw_layout.addWidget(QLabel("Model V8 :")); hw_layout.addWidget(self.cb_model)

        self.cb_res = QComboBox()
        self.lbl_warning = QLabel("Checking camera resulation")
        self.lbl_warning.setStyleSheet("color: #ff9933; font-size: 11px; font-style: italic;")
        hw_layout.addWidget(QLabel("Độ phân giải hỗ trợ:")); hw_layout.addWidget(self.cb_res)
        hw_layout.addWidget(self.lbl_warning)
        
        left_panel.addWidget(self.hw_widget) # Hardware load

        self.lbl_conf = QLabel("Chọn mức Confidence: ")
        self.sld_conf = QSlider(Qt.Orientation.Horizontal)
        self.sld_conf.setRange(20, 90); self.sld_conf.setValue(40)
        self.sld_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"Confidence: {v}%"))
        left_panel.addWidget(self.lbl_conf); left_panel.addWidget(self.sld_conf)

        self.chk_box = QCheckBox("Hiển thị khung "); self.chk_box.setChecked(True)
        self.chk_acc = QCheckBox("Hiển thị % accury"); self.chk_acc.setChecked(True)
        self.chk_fps = QCheckBox("Hiển thị FPS"); self.chk_fps.setChecked(True)
        left_panel.addWidget(self.chk_box); left_panel.addWidget(self.chk_acc); left_panel.addWidget(self.chk_fps)

        self.cb_lang = QComboBox(); self.cb_lang.addItems(["Tiếng Việt", "English"])
        left_panel.addWidget(QLabel("Chọn ngôn ngữ đọc:")); left_panel.addWidget(self.cb_lang)

        # start_stop
        self.btn_toggle = QPushButton("BẮT ĐẦU")
        self.btn_toggle.setFixedHeight(50)
        self.set_btn_style("start")
        self.btn_toggle.clicked.connect(self.toggle_app)
        left_panel.addWidget(self.btn_toggle)
        left_panel.addStretch()

        self.v_label = QLabel("HỆ THỐNG CHƯA CHẠY\nVui lòng cấu hình và bấm BẮT ĐẦU.")
        self.v_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.v_label.setStyleSheet("background-color: #000000; border: 2px dashed #555; color: #888; font-size: 16px;")
        self.v_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        main_layout.addWidget(side_bar)
        main_layout.addWidget(self.v_label, stretch=1)
        self.setLayout(main_layout)

    def set_btn_style(self, state):
        if state == "start":
            self.btn_toggle.setText("BẮT ĐẦU HOẠT ĐỘNG")
            self.btn_toggle.setStyleSheet("QPushButton { background-color: #2ecc71; color: black; font-weight: bold; border-radius: 8px; font-size: 14px;} QPushButton:hover { background-color: #27ae60; }")
        elif state == "stop":
            self.btn_toggle.setText("DỪNG")
            self.btn_toggle.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; border-radius: 8px; font-size: 14px;} QPushButton:hover { background-color: #c0392b; }")

    def scan_supported_resolutions(self):
        cam_id = self.cb_cam.currentData()
        if cam_id is None: return
        
        self.cb_res.clear()
        self.lbl_warning.setText("Scanning for resolution")
        QApplication.processEvents() 
        
        # list test
        res_to_test = [
            ("640x480 ", 640, 480),
            ("800x600 ", 800, 600),
            ("1024x768 ", 1024, 768),
            ("1280x720 ", 1280, 720),
            ("1280x1024 ", 1280, 1024),
            ("1600x900 ", 1600, 900),
            ("1920x1080 ", 1920, 1080),
            ("2560x1440 ", 2560, 1440)
        ]
        
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        supported_count = 0
        
        if cap.isOpened():
            for name, w, h in res_to_test:
                # camera cưỡng bức
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                
                #camera test
                actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                if actual_w == w and actual_h == h:
                    self.cb_res.addItem(name, (int(w), int(h)))
                    supported_count += 1
            cap.release()

        if supported_count > 0:
            self.lbl_warning.setText(f"Find {supported_count} resolution.")
            self.lbl_warning.setStyleSheet("color: #2ecc71; font-size: 11px;")
        else:
            self.cb_res.addItem("Mặc định (640x480)", (640, 480))
            self.lbl_warning.setText("Not found any!")

    def toggle_app(self):
        # logic start stop
        if not self.is_running:
            self.start_app()
        else:
            self.stop_app()

    def start_app(self):
        if self.cb_res.count() == 0: return
        self.is_running = True
        
        # lock settings
        self.hw_widget.setEnabled(False) 
        
        # switch on off
        self.btn_toggle.setText("AI Loading now...")
        self.btn_toggle.setEnabled(False)
        QApplication.processEvents()

        # model and camera
        res = self.cb_res.currentData()
        model_name = self.cb_model.currentText().split()[0]
        self.model = YOLO(model_name)
        self.cap = cv2.VideoCapture(self.cb_cam.currentData(), cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0]); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        
        self.btn_toggle.setEnabled(True)
        self.set_btn_style("stop")
        
        self.lang = "vi" if self.cb_lang.currentText() == "Tiếng Việt" else "en"
        self.last_speak = 0
        
        # Loop 
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5) 

    def stop_app(self):
        self.is_running = False
        
        # stop
        if hasattr(self, 'timer'): self.timer.stop()
        if hasattr(self, 'cap'): self.cap.release()
        
        # unlock
        self.hw_widget.setEnabled(True)
        
        # change
        self.set_btn_style("start")
        self.v_label.clear()
        self.v_label.setText("ĐÃ DỪNG \nBạn có thể chỉnh sửa cấu hình.")

    def speak(self, text):
        if self.is_speaking: return
        def run():
            self.is_speaking = True
            try:
                gTTS(text=text, lang=self.lang).save("v.mp3")
                pygame.mixer.music.load("v.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.is_running: time.sleep(0.1) # stop speaking
                pygame.mixer.music.unload()
                if os.path.exists("v.mp3"): os.remove("v.mp3")
            except: pass
            self.is_speaking = False
        threading.Thread(target=run, daemon=True).start()

    def update_frame(self):
        if not self.is_running: return # lock safe
        ret, frame = self.cap.read()
        if not ret: return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        conf_val = self.sld_conf.value() / 100
        
        results = self.model.predict(frame_rgb, conf=conf_val, imgsz=320, verbose=False)
        h, w, _ = frame.shape
        objs_found = []

        img_p = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_p)
        
        for r in results:
            sorted_boxes = sorted(r.boxes, key=lambda x: x.conf[0], reverse=True)
            for b in sorted_boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                eng_name = self.model.names[int(b.cls[0])]
                name = self.trans.get(eng_name, eng_name) if self.lang == "vi" else eng_name
                
                cx = (x1 + x2) / 2
                pos = "bên trái" if cx < w/3 else "ở giữa" if cx < 2*w/3 else "bên phải"
                if self.lang == "en": pos = "left" if cx < w/3 else "center" if cx < 2*w/3 else "right"
                
                objs_found.append({"name": name, "pos": pos})
                
                if self.chk_box.isChecked():
                    draw.rectangle([x1, y1, x2, y2], outline="#00ff00", width=3)
                    txt = f" {name} " + (f"| {int(conf*100)}%" if self.chk_acc.isChecked() else "")
                    draw.rectangle([x1, y1-30, x1+draw.textlength(txt, font=self.p_font)+5, y1], fill="#00ff00")
                    draw.text((x1, y1-30), txt, fill="black", font=self.p_font)

        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        if self.chk_fps.isChecked():
            draw.text((20, 20), f"FPS: {int(fps)}", fill="#ffff00", font=self.p_font)

        if time.time() - self.last_speak > 4 and objs_found:
            best_obj = objs_found[0]
            txt_speak = f"Phía trước có {best_obj['name']} {best_obj['pos']}" if self.lang == "vi" else f"I see a {best_obj['name']} on the {best_obj['pos']}"
            self.speak(txt_speak)
            self.last_speak = time.time()

        qt_img = QImage(np.array(img_p).data, w, h, 3*w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.v_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.v_label.setPixmap(scaled_pixmap)

    def closeEvent(self, e):
        self.is_running = False
        if hasattr(self, 'cap'): self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIVisionApp(); win.show()
    sys.exit(app.exec())