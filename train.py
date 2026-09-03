"""
YOLOv9 Çelik Yüzey Hatası Tespit Modeli - Eğitim Scripti
===========================================================
NEU-DET veri seti üzerinde YOLOv9 modelini eğitir.
Transfer learning yaklaşımı ile COCO üzerine eğitilmiş ağırlıklar kullanılır.

Yazar: Berkay
Tarih: 2024
"""

from ultralytics import YOLO
import os
import yaml
import torch

# ============================================================
# KONFİGÜRASYON
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(PROJECT_DIR, "datasets", "NEU-DET", "data.yaml")
RESULTS_DIR = os.path.join(PROJECT_DIR, "runs")

# Eğitim parametreleri
CONFIG = {
    "model": "yolov9c.pt",      # Önceden eğitilmiş model (COCO)
    "data": DATA_YAML,           # Veri seti konfigürasyonu
    "epochs": 50,                # Eğitim epoch sayısı
    "imgsz": 640,                # Görüntü boyutu (200→640 otomatik scale)
    "batch": 8,                  # Batch boyutu (CPU için uygun)
    "device": "mps" if torch.backends.mps.is_available() else "cpu",  # Apple Silicon GPU (varsa)
    "optimizer": "SGD",          # Optimizer
    "lr0": 0.01,                 # Başlangıç öğrenme oranı
    "lrf": 0.01,                 # Son öğrenme oranı faktörü
    "momentum": 0.937,           # SGD momentum
    "weight_decay": 0.0005,      # Ağırlık azaltma
    "warmup_epochs": 3.0,        # Warmup epoch sayısı
    "warmup_momentum": 0.8,      # Warmup momentum
    "patience": 10,              # Early stopping sabır değeri
    "project": RESULTS_DIR,      # Sonuç kayıt dizini
    "name": "steel_defect_yolov9",  # Deney ismi
    "exist_ok": True,            # Mevcut dizini üzerine yaz
    "pretrained": True,          # Transfer learning aktif
    "verbose": True,             # Detaylı log
    "save": True,                # Model kaydet
    "save_period": 10,           # Her 10 epoch'ta checkpoint
    "plots": True,               # Eğitim grafikleri oluştur
    "val": True,                 # Her epoch sonunda doğrulama
    
    # Data Augmentation
    "hsv_h": 0.015,              # Renk tonu augmentation
    "hsv_s": 0.7,                # Doygunluk augmentation
    "hsv_v": 0.4,                # Parlaklık augmentation
    "degrees": 0.0,              # Döndürme
    "translate": 0.1,            # Öteleme
    "scale": 0.5,                # Ölçekleme
    "shear": 0.0,                # Eğme
    "flipud": 0.5,               # Dikey çevirme
    "fliplr": 0.5,               # Yatay çevirme
    "mosaic": 1.0,               # Mozaik augmentation
    "mixup": 0.0,                # Mixup augmentation
}


def print_config():
    """Eğitim konfigürasyonunu yazdırır."""
    print("=" * 60)
    print("  YOLOv9 Çelik Yüzey Hatası Tespit - Eğitim")
    print("=" * 60)
    
    # Cihaz bilgisi
    if torch.backends.mps.is_available():
        device_info = "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        device_info = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device_info = "CPU"
    
    print(f"\n  🖥️  Cihaz: {device_info}")
    print(f"  📁 Veri seti: {DATA_YAML}")
    print(f"  🤖 Model: {CONFIG['model']}")
    print(f"  📊 Epoch: {CONFIG['epochs']}")
    print(f"  📦 Batch: {CONFIG['batch']}")
    print(f"  📐 Görüntü boyutu: {CONFIG['imgsz']}")
    print(f"  📈 Öğrenme oranı: {CONFIG['lr0']}")
    print(f"  ⏰ Patience: {CONFIG['patience']}")
    print(f"  💾 Sonuçlar: {RESULTS_DIR}")
    
    # Veri seti bilgisi
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"\n  Sınıf sayısı: {data_config['nc']}")
    print(f"  Sınıflar: {list(data_config['names'].values())}")
    print("=" * 60)


def main():
    print_config()
    
    print("\n  🚀 Model yükleniyor...")
    model = YOLO(CONFIG["model"])
    
    print("  🏋️ Eğitim başlıyor...\n")
    
    # Eğitim parametrelerini ayarla
    train_params = {k: v for k, v in CONFIG.items() if k != "model"}
    
    # Eğitimi başlat
    results = model.train(**train_params)
    
    print("\n" + "=" * 60)
    print("  ✅ Eğitim tamamlandı!")
    print(f"  📁 Sonuçlar: {RESULTS_DIR}/steel_defect_yolov9")
    print(f"  🏆 En iyi model: {RESULTS_DIR}/steel_defect_yolov9/weights/best.pt")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
