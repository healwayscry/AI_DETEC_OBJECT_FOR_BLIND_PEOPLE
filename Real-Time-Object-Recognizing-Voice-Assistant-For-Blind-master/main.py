import cv2
import time
import threading
import os
from ultralytics import YOLO
from gtts import gTTS
import pygame

# ===== INIT AUDIO =====
pygame.mixer.init()

def speak(text):
    def run():
        try:
            filename = "voice.mp3"

            tts = gTTS(text=text, lang='vi')
            tts.save(filename)

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.stop()
            pygame.mixer.music.unload()

            os.remove(filename)

        except:
            pass

    threading.Thread(target=run).start()

# ===== LOAD YOLOv8 =====
model = YOLO("yolov8n.pt")  

# ===== MAP TIẾNG VIỆT =====
label_map = {
    "person": "người",
    "cell phone": "điện thoại",
    "bottle": "chai",
    "chair": "ghế",
    "laptop": "máy tính",
    "car": "xe",
    "dog": "chó",
    "cat": "mèo"
}

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

last_speak_time = 0

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # ===== YOLOv8 DETECT =====
    results = model(frame, conf=0.4)

    detected_objects = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            detected_objects.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            text = f"{label} {int(conf*100)}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if time.time() - last_speak_time > 3:

        if detected_objects:
            vn_labels = [label_map.get(obj, obj) for obj in set(detected_objects)]
            voice_text = "Phía trước bạn có " + ", ".join(vn_labels)
        else:
            voice_text = "Phía trước bạn không có vật thể"

        print("VOICE:", voice_text)

        speak(voice_text)

        last_speak_time = time.time()

    end_time = time.time()
    print("FPS:", int(1/(end_time - start_time)))

cap.release()
cv2.destroyAllWindows()