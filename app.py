"""
YOLOv9 Çelik Yüzey Hatası Tespiti - Web Arayüzü
==================================================
Gradio ile eğitilmiş modeli tarayıcı üzerinden test etmek için basit bir arayüz.
"""

import os
import glob
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "runs", "steel_defect_yolov9", "weights", "best.pt")
VAL_IMG_DIR = os.path.join(PROJECT_DIR, "datasets", "NEU-DET", "val", "images")

CLASS_NAMES_TR = {
    0: "Çatlama", 1: "Kapanma", 2: "Yamalar",
    3: "Çukurlu Yüzey", 4: "Hadde Tufali", 5: "Çizikler",
}
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

model = YOLO(MODEL_PATH)


def predict(image, conf_threshold):
    if image is None:
        return None, "Lütfen bir görsel yükleyin."

    result = model.predict(source=image, imgsz=640, conf=conf_threshold, verbose=False)[0]

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    lines = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        color = COLORS[cls_id % len(COLORS)]
        name = CLASS_NAMES_TR.get(cls_id, f"Sınıf {cls_id}")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{name} {conf:.2f}"
        text_y = max(0, y1 - 12)
        draw.rectangle([x1, text_y, x1 + 7 * len(label), text_y + 12], fill=color)
        draw.text((x1 + 1, text_y), label, fill="black")

        lines.append(f"• {name} — güven: {conf:.1%}")

    summary = "\n".join(lines) if lines else "Hiçbir yüzey hatası tespit edilmedi."
    return annotated, summary


def example_images():
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    examples = []
    for cls_name in class_names:
        matches = sorted(glob.glob(os.path.join(VAL_IMG_DIR, f"{cls_name}_*.jpg")))
        if matches:
            examples.append([matches[0], 0.25])
    return examples


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Çelik Yüzey Görseli"),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Güven Eşiği (Confidence)"),
    ],
    outputs=[
        gr.Image(type="pil", label="Tespit Sonucu"),
        gr.Textbox(label="Tespit Edilen Hatalar", lines=6),
    ],
    title="🔩 Çelik Yüzey Hatası Tespiti — YOLOv9",
    description=(
        "NEU Surface Defect Database üzerinde eğitilmiş YOLOv9c modeli. "
        "Bir çelik yüzey görseli yükleyin, model hataları kutulayıp sınıflandırsın."
    ),
    examples=example_images(),
)

if __name__ == "__main__":
    demo.launch()
