import sys, cv2, time, threading, os, pygame, json, numpy as np
from ultralytics import YOLO
from gtts import gTTS
from PIL import ImageFont, ImageDraw, Image
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

# CLASS DIALOG
class ClassSelectionDialog(QDialog):
    def __init__(self, names_dict, trans_dict, current_selection, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chọn Vật Thể Cần Nhận Diện")
        self.resize(400, 500)
        self.setStyleSheet("background-color: #2b2b2b; color: #f0f0f0; font-family: Arial;")
        
        layout = QVBoxLayout(self)
        
        # CHECK BOX
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("background-color: #1a1a1a; border: 1px solid #555; padding: 5px;")
        
        for cid, eng in names_dict.items():
            name_vi = trans_dict.get(eng, eng)
            item = QListWidgetItem(f"{name_vi} ({eng})")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            
            # khôi phục static
            if cid in current_selection: item.setCheckState(Qt.CheckState.Checked)
            else: item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, cid)
            self.list_widget.addItem(item)
            
        layout.addWidget(self.list_widget)
        
        # button
        btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("Chọn tất cả")
        btn_deselect_all = QPushButton("Bỏ chọn hết")
        btn_save = QPushButton("Lưu")
        
        # styling
        btn_save.setStyleSheet("background-color: #2ecc71; color: black; font-weight: bold;")
        
        btn_select_all.clicked.connect(self.select_all)
        btn_deselect_all.clicked.connect(self.deselect_all)
        btn_save.clicked.connect(self.accept) # dong va tra ve ma accepted
        
        btn_layout.addWidget(btn_select_all)
        btn_layout.addWidget(btn_deselect_all)
        btn_layout.addWidget(btn_save)
        
        layout.addLayout(btn_layout)
        
    def select_all(self):
        for i in range(self.list_widget.count()): self.list_widget.item(i).setCheckState(Qt.CheckState.Checked)
            
    def deselect_all(self):
        for i in range(self.list_widget.count()): self.list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)
            
    def get_selected(self):
        # list item checked
        return [self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) 
                for i in range(self.list_widget.count()) 
                if self.list_widget.item(i).checkState() == Qt.CheckState.Checked]

# MAIN CLASS APP AI
class AIVisionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Hỗ Trợ Nhận Diện Vật Thể")
        self.setStyleSheet("background-color: #1a1a1a; color: #f0f0f0; font-family: Arial;")
        self.resize(1280, 720)
        
        # Load EN -> VN
        self.trans = {}
        if os.path.exists("translations.json"):
            with open("translations.json", "r", encoding="utf-8") as f:
                self.trans = json.load(f)
        
        # Load font
        try: self.p_font = ImageFont.truetype("arial.ttf", 22)
        except: self.p_font = ImageFont.load_default()

        # LOAD MODEL NAME FROM YOLO
        temp_model = YOLO("yolov8n.pt") 
        self.all_class_names = temp_model.names
        self.selected_classes_gui = list(self.all_class_names.keys()) # DEFAULT: check all
        
        # Library speak
        pygame.mixer.init()
        self.is_speaking = False
        self.is_running = False 
        self.prev_time = 0
        
        self.init_ui()
        self.scan_supported_resolutions()

    def get_cams(self):
        # scanning camera
        arr = []
        for i in range(2):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened(): arr.append(i); cap.release()
        return arr

    def init_ui(self):
        # GRID LEFT RIGHT MIDDLE
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

        #GRID CHOOSE CAMERA
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

        # CAMERA RESULATION
        self.cb_res = QComboBox()
        self.lbl_warning = QLabel("Checking camera resulation")
        self.lbl_warning.setStyleSheet("color: #ff9933; font-size: 11px; font-style: italic;")
        hw_layout.addWidget(QLabel("Độ phân giải hỗ trợ:")); hw_layout.addWidget(self.cb_res)
        hw_layout.addWidget(self.lbl_warning)
        
        left_panel.addWidget(self.hw_widget) 

        # (Confidence)
        self.lbl_conf = QLabel("Chọn mức Confidence: ")
        self.sld_conf = QSlider(Qt.Orientation.Horizontal)
        self.sld_conf.setRange(20, 90); self.sld_conf.setValue(40)
        self.sld_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"Confidence: {v}%"))
        left_panel.addWidget(self.lbl_conf); left_panel.addWidget(self.sld_conf)

        # 3
        self.chk_box = QCheckBox("Hiển thị khung "); self.chk_box.setChecked(True)
        self.chk_acc = QCheckBox("Hiển thị % accury"); self.chk_acc.setChecked(True)
        self.chk_fps = QCheckBox("Hiển thị FPS"); self.chk_fps.setChecked(True)
        self.chk_dist = QCheckBox("Cảnh báo xa/gần"); self.chk_dist.setChecked(True)
        left_panel.addWidget(self.chk_box); left_panel.addWidget(self.chk_acc); left_panel.addWidget(self.chk_fps); left_panel.addWidget(self.chk_dist)

        # 4
        self.btn_select_classes = QPushButton("Cài đặt Vật thể nhận diện...")
        self.btn_select_classes.setStyleSheet("background-color: #3498db; color: white; border-radius: 5px; padding: 8px;")
        self.btn_select_classes.clicked.connect(self.open_class_dialog)
        left_panel.addWidget(self.btn_select_classes)

        # 5
        self.cb_lang = QComboBox(); self.cb_lang.addItems(["Tiếng Việt", "English"])
        left_panel.addWidget(QLabel("Chọn ngôn ngữ đọc:")); left_panel.addWidget(self.cb_lang)

        self.btn_toggle = QPushButton("BẮT ĐẦU")
        self.btn_toggle.setFixedHeight(50)
        self.set_btn_style("start")
        self.btn_toggle.clicked.connect(self.toggle_app)
        left_panel.addWidget(self.btn_toggle)
        left_panel.addStretch()

        # CAMERA GRID
        self.v_label = QLabel("HỆ THỐNG CHƯA CHẠY\nVui lòng cấu hình và bấm BẮT ĐẦU.")
        self.v_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.v_label.setStyleSheet("background-color: #000000; border: 2px dashed #555; color: #888; font-size: 16px;")
        self.v_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        main_layout.addWidget(side_bar)
        main_layout.addWidget(self.v_label, stretch=1)
        self.setLayout(main_layout)

    def open_class_dialog(self):
        #popup choose
        dlg = ClassSelectionDialog(self.all_class_names, self.trans, self.selected_classes_gui, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.selected_classes_gui = dlg.get_selected() # update list
    def set_btn_style(self, state):
        # change color on off
        if state == "start":
            self.btn_toggle.setText("BẮT ĐẦU HOẠT ĐỘNG")
            self.btn_toggle.setStyleSheet("QPushButton { background-color: #2ecc71; color: black; font-weight: bold; border-radius: 8px; font-size: 14px;} QPushButton:hover { background-color: #27ae60; }")
        elif state == "stop":
            self.btn_toggle.setText("DỪNG")
            self.btn_toggle.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-weight: bold; border-radius: 8px; font-size: 14px;} QPushButton:hover { background-color: #c0392b; }")

    def scan_supported_resolutions(self):
        # scanning for res
        cam_id = self.cb_cam.currentData()
        if cam_id is None: return
        
        self.cb_res.clear()
        self.lbl_warning.setText("Scanning for resolution")
        QApplication.processEvents() 
        
        res_to_test = [
            ("640x480 ", 640, 480), ("800x600 ", 800, 600), ("1024x768 ", 1024, 768),
            ("1280x720 ", 1280, 720), ("1280x1024 ", 1280, 1024), ("1600x900 ", 1600, 900),
            ("1920x1080 ", 1920, 1080), ("2560x1440 ", 2560, 1440)
        ]
        
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        supported_count = 0
        
        if cap.isOpened():
            for name, w, h in res_to_test:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                
                if cap.get(cv2.CAP_PROP_FRAME_WIDTH) == w and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == h:
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
        if not self.is_running: self.start_app()
        else: self.stop_app()

    def start_app(self):
        if self.cb_res.count() == 0: return
        self.is_running = True
        
        # lock frontend
        self.hw_widget.setEnabled(False) 
        self.btn_select_classes.setEnabled(False)
        self.btn_toggle.setText("AI Loading now...")
        self.btn_toggle.setEnabled(False)
        QApplication.processEvents()

        # apply filter
        if len(self.selected_classes_gui) == len(self.all_class_names):
            self.selected_classes = None # None = Nhận diện tất cả
        elif len(self.selected_classes_gui) == 0:
            self.selected_classes = [-1] # Mảng rác = Không nhận diện gì
        else:
            self.selected_classes = self.selected_classes_gui

        # run model ai and camera
        res = self.cb_res.currentData()
        model_name = self.cb_model.currentText().split()[0]
        self.model = YOLO(model_name)
        self.cap = cv2.VideoCapture(self.cb_cam.currentData(), cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0]); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        
        self.btn_toggle.setEnabled(True)
        self.set_btn_style("stop")
        
        self.lang = "vi" if self.cb_lang.currentText() == "Tiếng Việt" else "en"
        self.last_speak = 0
        
        # loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5) 

    def stop_app(self):
        # stop thread
        self.is_running = False
        if hasattr(self, 'timer'): self.timer.stop()
        if hasattr(self, 'cap'): self.cap.release()
        
        # unlock setting
        self.hw_widget.setEnabled(True)
        self.btn_select_classes.setEnabled(True)
        self.set_btn_style("start")
        self.v_label.clear()
        self.v_label.setText("ĐÃ DỪNG \nBạn có thể chỉnh sửa cấu hình.")

    def speak(self, text):
        # multi thread for high fps
        if self.is_speaking: return
        def run():
            self.is_speaking = True
            try:
                gTTS(text=text, lang=self.lang).save("v.mp3") # save to mp3
                pygame.mixer.music.load("v.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.is_running: time.sleep(0.1) # wating
                pygame.mixer.music.unload()
                if os.path.exists("v.mp3"): os.remove("v.mp3") # delete mp3 
            except: pass
            self.is_speaking = False
        threading.Thread(target=run, daemon=True).start()

    def update_frame(self):
        # xử lý khung hình camera liên tục
        if not self.is_running: return 
        ret, frame = self.cap.read()
        if not ret: return
        
        # switch to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        conf_val = self.sld_conf.value() / 100
        
        if self.selected_classes == [-1]: results = []
        else: results = self.model.predict(frame_rgb, conf=conf_val, imgsz=320, classes=self.selected_classes, verbose=False)
            
        h, w, _ = frame.shape
        objs_found = []

        # pillow
        img_p = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_p)
        
        for r in results:
            sorted_boxes = sorted(r.boxes, key=lambda x: x.conf[0], reverse=True)
            for b in sorted_boxes:
                # Trích xuất dữ liệu: Tọa độ, Confidence, Tên vật thể
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                eng_name = self.model.names[int(b.cls[0])]
                name = self.trans.get(eng_name, eng_name) if self.lang == "vi" else eng_name
                
                # position 
                cx = (x1 + x2) / 2
                pos = "bên trái" if cx < w/3 else "ở giữa" if cx < 2*w/3 else "bên phải"
                if self.lang == "en": pos = "left" if cx < w/3 else "center" if cx < 2*w/3 else "right"
                
                # Calculate distance
                dist_str_draw, dist_str_speak = "", ""
                draw_color = "#00ff00"
                if self.chk_dist.isChecked():
                    box_h = y2 - y1
                    if box_h / h > 0.6: 
                        dist_str_draw = " | Rất gần" if self.lang == "vi" else " | Very close"
                        dist_str_speak = "ở rất gần" if self.lang == "vi" else "very close"
                        draw_color = "#ff0000"
                    elif box_h / h > 0.3: 
                        dist_str_draw = " | Gần" if self.lang == "vi" else " | Close"
                        dist_str_speak = "ở gần" if self.lang == "vi" else "close"
                        draw_color = "#ffa500"
                    else: 
                        dist_str_draw = " | Xa" if self.lang == "vi" else " | Far"
                        dist_str_speak = "ở xa" if self.lang == "vi" else "far"

                objs_found.append({"name": name, "pos": pos, "dist": dist_str_speak})
                
                #bounding box
                if self.chk_box.isChecked():
                    draw.rectangle([x1, y1, x2, y2], outline=draw_color, width=3)
                    txt = f" {name} " + (f"| {int(conf*100)}%" if self.chk_acc.isChecked() else "") + dist_str_draw
                    draw.rectangle([x1, y1-30, x1+draw.textlength(txt, font=self.p_font)+5, y1], fill=draw_color)
                    draw.text((x1, y1-30), txt, fill="black", font=self.p_font)

        # fps calculate
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        if self.chk_fps.isChecked():
            draw.text((20, 20), f"FPS: {int(fps)}", fill="#ffff00", font=self.p_font)

        # 4 sec 1 speak
        if time.time() - self.last_speak > 4 and objs_found:
            best_obj = objs_found[0]
            if best_obj['dist']: txt_speak = f"Phía trước có {best_obj['name']} {best_obj['pos']} {best_obj['dist']}" if self.lang == "vi" else f"I see a {best_obj['name']} on the {best_obj['pos']}, it is {best_obj['dist']}"
            else: txt_speak = f"Phía trước có {best_obj['name']} {best_obj['pos']}" if self.lang == "vi" else f"I see a {best_obj['name']} on the {best_obj['pos']}"
            
            self.speak(txt_speak)
            self.last_speak = time.time()

        # Chuyển đổi khung hình về dạng PyQt 
        qt_img = QImage(np.array(img_p).data, w, h, 3*w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.v_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.v_label.setPixmap(scaled_pixmap)

    def closeEvent(self, e):
        # safe lock
        self.is_running = False
        if hasattr(self, 'cap'): self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIVisionApp(); win.show()
    sys.exit(app.exec())