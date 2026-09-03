"""
YOLOv9 Çelik Yüzey Hatası Tespit - Tahmin Scripti
====================================================
Eğitilmiş model ile tek görsel veya klasör üzerinde
tahmin yapar. Sonuçlar bounding box'larla kaydedilir.
"""

from ultralytics import YOLO
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "runs", "steel_defect_yolov9")
SCREENSHOT_DIR = os.path.join(PROJECT_DIR, "screenshots")
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "NEU-DET")

CLASS_NAMES_TR = {
    0: "Çatlama", 1: "Kapanma", 2: "Yamalar",
    3: "Çukurlu Yüzey", 4: "Hadde Tufali", 5: "Çizikler"
}

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def predict_samples():
    """Her sınıftan doğrulama setinden örnekler üzerinde tahmin yapar."""
    best_model_path = os.path.join(RESULTS_DIR, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        print(f"  ❌ Model bulunamadı: {best_model_path}")
        return
    
    model = YOLO(best_model_path)
    val_img_dir = os.path.join(DATASET_DIR, "val", "images")
    
    # Her sınıftan 1 örnek seç
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    sample_images = []
    
    for cls_name in class_names:
        # Doğrulama setinden bir görsel bul
        pattern = os.path.join(val_img_dir, f"{cls_name}_*.jpg")
        matches = sorted(glob.glob(pattern))
        if matches:
            sample_images.append(matches[0])
    
    if not sample_images:
        print("  ❌ Örnek görsel bulunamadı!")
        return
    
    print(f"\n  🔮 {len(sample_images)} görsel üzerinde tahmin yapılıyor...\n")
    
    # Tahminleri yap
    results = model.predict(source=sample_images, imgsz=640, conf=0.25, 
                           save=False, verbose=False)
    
    # Sonuçları görselleştir
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLOv9 Çelik Yüzey Hatası Tespiti - Tahmin Sonuçları', 
                 fontsize=20, fontweight='bold')
    
    for idx, (result, img_path) in enumerate(zip(results, sample_images)):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        
        # Orijinal görseli göster
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        
        # Tahmin edilen bounding box'ları çiz
        boxes = result.boxes
        detected_count = 0
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            color = COLORS[cls_id] if cls_id < len(COLORS) else '#FF0000'
            cls_name_tr = CLASS_NAMES_TR.get(cls_id, f"Sınıf {cls_id}")
            
            # Box çiz
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor=color, alpha=0.15
            )
            ax.add_patch(rect)
            rect_border = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect_border)
            
            # Etiket yaz
            ax.text(x1, y1 - 3, f'{cls_name_tr} {conf:.2f}', 
                   fontsize=8, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
            detected_count += 1
        
        # Dosya adından gerçek sınıfı çıkar
        basename = os.path.basename(img_path)
        true_class = '_'.join(basename.split('_')[:-1])
        true_class_tr = CLASS_NAMES_TR.get(
            list(CLASS_NAMES_TR.keys())[list(
                ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
            ).index(true_class)] if true_class in ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'] else 0, 
            true_class
        )
        
        ax.set_title(f'Gerçek: {true_class_tr}\nTespit: {detected_count} nesne', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(SCREENSHOT_DIR, "14_tahmin_sonuclari.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Tahmin sonuçları kaydedildi: {save_path}")


def predict_grid():
    """Daha fazla örnek üzerinde büyük bir grid tahmin görseli oluşturur."""
    best_model_path = os.path.join(RESULTS_DIR, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        return
    
    model = YOLO(best_model_path)
    val_img_dir = os.path.join(DATASET_DIR, "val", "images")
    
    # Her sınıftan 3'er örnek seç
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    fig.suptitle('YOLOv9 - Tüm Sınıflardan Tahmin Örnekleri', fontsize=22, fontweight='bold')
    
    for cls_idx, cls_name in enumerate(class_names):
        pattern = os.path.join(val_img_dir, f"{cls_name}_*.jpg")
        matches = sorted(glob.glob(pattern))[:3]
        
        for sample_idx, img_path in enumerate(matches):
            ax = axes[cls_idx][sample_idx]
            
            img = Image.open(img_path)
            result = model.predict(source=img_path, imgsz=640, conf=0.25, verbose=False)[0]
            
            ax.imshow(img, cmap='gray')
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                color = COLORS[cls_id] if cls_id < len(COLORS) else '#FF0000'
                
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 2, f'{conf:.2f}', fontsize=7, color='white',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.8))
            
            if sample_idx == 0:
                ax.set_ylabel(CLASS_NAMES_TR[cls_idx], fontsize=12, fontweight='bold', rotation=0,
                            labelpad=80, va='center')
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(SCREENSHOT_DIR, "15_tahmin_grid.png")
    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Tahmin grid kaydedildi: {save_path}")


def main():
    print("=" * 60)
    print("  YOLOv9 Çelik Yüzey Hatası Tespit - Tahmin")
    print("=" * 60)
    
    predict_samples()
    predict_grid()
    
    print(f"\n  ✅ Tahminler tamamlandı!")
    print(f"  📁 Görseller: {SCREENSHOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
