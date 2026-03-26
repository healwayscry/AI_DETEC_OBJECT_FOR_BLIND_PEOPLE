import cv2
import time
import threading
import os
import json
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import pygame
from PIL import ImageFont, ImageDraw, Image
from collections import Counter

# ================== CẤU HÌNH NÂNG CAO ==================
TRANSLATION_FILE = "translations.json"
# Đổi sang 'yolov8s.pt' để thông minh hơn, hoặc 'yolov8m.pt' nếu máy có GPU mạnh
MODEL_PATH = "yolov8s.pt" 
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 

model = YOLO(MODEL_PATH)

# Bộ nhớ đệm để lọc nhiễu (Smoothing)
object_history = [] 
HISTORY_LIMIT = 5 # Phải xuất hiện 5 lần mới tin là có thật

translation_cache = {}
if os.path.exists(TRANSLATION_FILE):
    with open(TRANSLATION_FILE, "r", encoding="utf-8") as f:
        translation_cache = json.load(f)

# ================== HÀM VẼ TỐI ƯU ==================
def draw_text_vietnamese(img_rgb, text, position, font, color=(0, 255, 0)):
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ================== ÂM THANH ==================
pygame.mixer.init()
is_speaking = False

def speak(text):
    global is_speaking
    if is_speaking: return
    def run():
        global is_speaking
        is_speaking = True
        try:
            tts = gTTS(text=text, lang='vi')
            tts.save("voice.mp3")
            pygame.mixer.music.load("voice.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
            pygame.mixer.music.unload()
            if os.path.exists("voice.mp3"): os.remove("voice.mp3")
        except: pass
        is_speaking = False
    threading.Thread(target=run, daemon=True).start()

# ================== CAMERA & XỬ LÝ ==================
cap = cv2.VideoCapture(0)
# Để nhận diện chuẩn, camera nên để độ phân giải tốt
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_speak_time = 0
prev_time = 0
try:
    font_pill = ImageFont.truetype(FONT_PATH, 20)
except:
    font_pill = ImageFont.load_default()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # NÂNG CẤP: imgsz=640 giúp AI nhìn chi tiết hơn rất nhiều
    results = model(frame, conf=0.45, imgsz=640, stream=True, verbose=False)
    
    current_frame_info = []
    frame_objects = []
    h_img, w_img, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_eng = model.names[cls_id]
            label_vn = translation_cache.get(label_eng, label_eng)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            
            # Tính vị trí
            if x_center < w_img // 3: pos = "trái"
            elif x_center < 2 * (w_img // 3): pos = "giữa"
            else: pos = "phải"

            # Lưu vào danh sách tạm của frame này
            obj_str = f"{label_vn} {pos}"
            frame_objects.append(obj_str)

            # Vẽ (Màu sắc theo độ tin cậy)
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            frame_rgb = draw_text_vietnamese(frame_rgb, f"{label_vn} {conf:.2f}", (x1, y1 - 25), font_pill, color)

    # NÂNG CẤP: Chỉ xác nhận vật thể nếu nó xuất hiện ổn định (Smoothing)
    object_history.append(frame_objects)
    if len(object_history) > HISTORY_LIMIT:
        object_history.pop(0)

    # Đếm xem vật thể nào xuất hiện nhiều nhất trong 5 frame gần đây
    flat_history = [item for sublist in object_history for item in sublist]
    stable_objects = [obj for obj, count in Counter(flat_history).items() if count >= 3]

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(frame, f"FPS: {int(fps)} | Model: {MODEL_PATH}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # LOGIC VOICE
    if time.time() - last_speak_time > 4:
        if stable_objects:
            msg = "Phía trước có " + " và ".join(stable_objects[:2])
            speak(msg)
            last_speak_time = time.time()

    cv2.imshow("AI Vision Ultra - High Accuracy", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()