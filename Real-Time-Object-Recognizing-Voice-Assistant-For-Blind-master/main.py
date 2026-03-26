import cv2
import time
import threading
import os
import json
from ultralytics import YOLO
from gtts import gTTS
import pygame
from transformers import MarianMTModel, MarianTokenizer

# ================== CẤU HÌNH FILE LƯU TRỮ ==================
TRANSLATION_FILE = "translations.json"
MODEL_PATH = "runs/detect/train/weights/best.pt"

# Load YOLO
model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else YOLO("yolov8n.pt")

#Important
label_map = {
    "person": "người", "cell phone": "điện thoại", "bottle": "chai nước",
    "chair": "cái ghế", "laptop": "máy tính", "car": "xe ô tô"
}

translation_cache = {}

# ================== LOGIC LOAD/SAVE BẢN DỊCH ==================
def load_translations():
    if os.path.exists(TRANSLATION_FILE):
        with open(TRANSLATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_translations(data):
    with open(TRANSLATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ================== KHỞI TẠO AI DỊCH (CHỈ LOAD 1 LẦN) ==================
print("Đang khởi động bộ não AI dịch (vui lòng đợi giây lát)...")
model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name)

def translate_with_ai(english_text):
    """Hàm này giờ chỉ thực hiện dịch, không load lại model nữa"""
    try:
        inputs = tokenizer(english_text.lower(), return_tensors="pt", padding=True)
        translated = translator_model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        return result.lower().replace("một ", "").strip()
    except:
        return english_text

# Sau đó mới đến vòng lặp kiểm tra và dịch labels
translation_cache = load_translations()
needs_save = False

for idx, name in model.names.items():
    if name not in translation_cache:
        if name in label_map:
            translation_cache[name] = label_map[name]
        else:
            print(f"-> Đang dịch từ mới: {name}")
            translation_cache[name] = translate_with_ai(name)
        needs_save = True

if needs_save:
    save_translations(translation_cache)
    print("Đã lưu bản dịch mới vào file!")
else:
    print("Đã tải bản dịch từ file thành công.")

# ================== ÂM THANH & CAMERA (Giữ nguyên logic cũ) ==================
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

cap = cv2.VideoCapture(0)
last_speak_time = 0
last_objects = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, conf=0.5, imgsz=416, verbose=False)
    current_frame_labels = []

    for r in results:
        for box in r.boxes:
            label_eng = model.names[int(box.cls[0])]
            label_vn = translation_cache.get(label_eng, label_eng)
            current_frame_labels.append(label_vn)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_vn, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if time.time() - last_speak_time > 3:
        unique_objects = list(set(current_frame_labels))
        if unique_objects and set(unique_objects) != last_objects:
            speak("Phía trước có " + ", ".join(unique_objects[:3]))
            last_objects = set(unique_objects)
            last_speak_time = time.time()

    cv2.imshow("AI Vision Optimized", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()