import os
import json
from ultralytics import YOLO
from transformers import MarianMTModel, MarianTokenizer


MODEL_PATH = "runs/detect/train/weights/best.pt" 
TRANSLATION_FILE = "translations.json"

label_map = {
    "person": "người", 
    "cell phone": "điện thoại", 
    "bottle": "chai nước",
    "chair": "cái ghế", 
    "laptop": "máy tính", 
    "car": "xe ô tô",
    "motorcycle": "xe máy",
    "bus": "xe buýt",
    "truck": "xe tải"
}

model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name)

def translate_with_ai(english_text):
    try:
        text = english_text.replace("-", " ").replace("_", " ").lower()
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = translator_model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        final = result.lower().replace("một ", "").strip()
        return final
    except Exception as e:
        print(f"Lỗi khi dịch từ '{english_text}': {e}")
        return english_text

# ================== QUY TRÌNH XỬ LÝ ==================
#load model
yolo_model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else YOLO("yolov8n.pt")
yolo_names = yolo_model.names

#load history word
if os.path.exists(TRANSLATION_FILE):
    with open(TRANSLATION_FILE, "r", encoding="utf-8") as f:
        translation_cache = json.load(f)
else:
    translation_cache = {}

print(f"Tìm thấy {len(yolo_names)} vật thể cần kiểm tra.")

# translate
needs_save = False
for idx, eng_name in yolo_names.items():
    if eng_name not in translation_cache:
        print(f"[*] Đang xử lý: '{eng_name}'", end=" -> ", flush=True)
        
        if eng_name in label_map:
            translation_cache[eng_name] = label_map[eng_name]
            print(f"{label_map[eng_name]} (Từ bản đồ tay)")
        else:
            viet_name = translate_with_ai(eng_name)
            translation_cache[eng_name] = viet_name
            print(f"{viet_name} (AI dịch)")
        
        needs_save = True

# Save result
if needs_save:
    with open(TRANSLATION_FILE, "w", encoding="utf-8") as f:
        json.dump(translation_cache, f, ensure_ascii=False, indent=4)
    print("---")
    print(f"HOÀN THÀNH! Đã lưu {len(translation_cache)} bản dịch vào {TRANSLATION_FILE}")
else:
    print("---")
    print("Không có từ nào mới")
