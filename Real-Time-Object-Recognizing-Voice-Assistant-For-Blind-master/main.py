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

# ================== CẤU HÌNH TỐI ƯU CPU ==================
TRANSLATION_FILE = "translations.json"
MODEL_PATH = "yolov8n.pt" 
FONT_PATH = "C:/Windows/Fonts/arial.ttf" 

# Chỉ dùng CPU
model = YOLO(MODEL_PATH)

translation_cache = {}
if os.path.exists(TRANSLATION_FILE):
    with open(TRANSLATION_FILE, "r", encoding="utf-8") as f:
        translation_cache = json.load(f)

# ================== LUỒNG CAMERA ==================
class WebcamStream:
    def __init__(self, src=1):
        self.cap = cv2.VideoCapture(src)
        # Tăng độ phân giải hiển thị cho đỡ "bé"
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.frame = frame

    def read(self):
        with self.read_lock:
            return self.grabbed, self.frame

    def stop(self):
        self.started = False
        self.cap.release()

# ================== HÀM VẼ (GIẢM TẢI) ==================
def draw_results(frame, results, font, translation_cache):
    frame_objects = []
    h_img, w_img = frame.shape[:2]
    
    # Chỉ chuyển sang PIL một lần duy nhất nếu có vật thể
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    has_obj = False

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.35: continue # Hạ thấp ngưỡng một chút để nhận diện được "nhiều đồ" hơn
            
            has_obj = True
            cls_id = int(box.cls[0])
            label_vn = translation_cache.get(model.names[cls_id], model.names[cls_id])
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            pos = "trái" if x_center < w_img // 3 else "giữa" if x_center < 2 * (w_img // 3) else "phải"
            frame_objects.append(f"{label_vn} {pos}")

            color = (0, 255, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1, y1 - 30), f"{label_vn}", font=font, fill=color)

    if has_obj:
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), frame_objects
    return frame, frame_objects

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

# ================== MAIN ==================
vs = WebcamStream(src=1).start()
time.sleep(2.0)

try:
    font_pill = ImageFont.truetype(FONT_PATH, 25)
except:
    font_pill = ImageFont.load_default()

prev_time = 0
last_speak_time = 0
frame_skip = 2 # SKIP FRAME: Chỉ chạy AI mỗi 2 khung hình để tăng FPS
count = 0

while True:
    ret, frame = vs.read()
    if not ret: break
    count += 1

    # Chỉ chạy AI mỗi n frame, các frame còn lại chỉ hiển thị hình ảnh
    if count % frame_skip == 0:
        # TỐI ƯU: imgsz=320 để CPU chạy mượt, nhưng vẽ lên frame to (1280x720)
        results = model.predict(frame, conf=0.35, imgsz=320, verbose=False)
        frame, frame_objects = draw_results(frame, results, font_pill, translation_cache)
        
        # Logic tiếng nói
        if time.time() - last_speak_time > 4 and frame_objects:
            speak("Phía trước có " + frame_objects[0])
            last_speak_time = time.time()

    # Tính FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("AI Vision - CPU Optimized", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

vs.stop()
cv2.destroyAllWindows()