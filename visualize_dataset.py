"""
NEU-DET Veri Seti Görselleştirme & İstatistik Raporu
=====================================================
Bu script, veri setinin istatistiklerini ve örnek görselleri
görselleştirerek staj defteri için SS'ler üretir.

Çıktılar screenshots/ klasörüne kaydedilir.
"""

import os
import glob
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalış
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from collections import Counter

# Proje kök dizini
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "NEU-DET")
SCREENSHOT_DIR = os.path.join(PROJECT_DIR, "screenshots")

# Sınıf bilgileri
CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches"
}

CLASS_NAMES_TR = {
    0: "Çatlama",
    1: "Kapanma",
    2: "Yamalar",
    3: "Çukurlu Yüzey",
    4: "Hadde Tufali",
    5: "Çizikler"
}

# Renk paleti
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def plot_class_distribution():
    """Sınıf dağılımı bar grafiği oluşturur."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('NEU-DET Veri Seti - Sınıf Dağılımı', fontsize=18, fontweight='bold', y=1.02)
    
    for idx, split in enumerate(['train', 'val']):
        label_dir = os.path.join(DATASET_DIR, split, "labels")
        class_counts = Counter()
        total_objects = 0
        
        if os.path.exists(label_dir):
            for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_objects += 1
        
        names = [CLASS_NAMES_TR[i] for i in range(6)]
        counts = [class_counts.get(i, 0) for i in range(6)]
        
        bars = axes[idx].bar(names, counts, color=COLORS, edgecolor='white', linewidth=2)
        axes[idx].set_title(f'{"Eğitim" if split == "train" else "Doğrulama"} Seti\n({sum(counts)} nesne)', 
                           fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Nesne Sayısı', fontsize=12)
        axes[idx].set_xlabel('Hata Sınıfı', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=30)
        
        # Bar üzerine sayı yaz
        for bar, count in zip(bars, counts):
            axes[idx].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 3,
                          str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(SCREENSHOT_DIR, "01_sinif_dagilimi.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Sınıf dağılımı kaydedildi: {save_path}")


def plot_sample_images():
    """Her sınıftan örnek görseller gösterir."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('NEU-DET - Her Sınıftan Örnek Görseller', fontsize=18, fontweight='bold')
    
    train_img_dir = os.path.join(DATASET_DIR, "train", "images")
    train_lbl_dir = os.path.join(DATASET_DIR, "train", "labels")
    
    for class_id in range(6):
        row, col = class_id // 3, class_id % 3
        ax = axes[row][col]
        
        class_name = CLASS_NAMES[class_id]
        # Bu sınıfa ait bir görsel bul
        pattern = os.path.join(train_img_dir, f"{class_name}_1.jpg")
        if os.path.exists(pattern):
            img = Image.open(pattern)
            ax.imshow(img, cmap='gray')
            
            # Bounding box'ları çiz
            label_file = os.path.join(train_lbl_dir, f"{class_name}_1.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            _, xc, yc, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            img_w, img_h = img.size
                            x1 = (xc - w/2) * img_w
                            y1 = (yc - h/2) * img_h
                            bw = w * img_w
                            bh = h * img_h
                            rect = mpatches.Rectangle((x1, y1), bw, bh, 
                                                      linewidth=2, edgecolor=COLORS[class_id], 
                                                      facecolor='none')
                            ax.add_patch(rect)
        
        ax.set_title(f'{CLASS_NAMES_TR[class_id]}\n({class_name})', fontsize=13, 
                    fontweight='bold', color=COLORS[class_id])
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(SCREENSHOT_DIR, "02_ornek_gorseller.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Örnek görseller kaydedildi: {save_path}")


def plot_bbox_analysis():
    """Bounding box boyut analizi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Bounding Box Boyut Analizi', fontsize=16, fontweight='bold')
    
    widths = {i: [] for i in range(6)}
    heights = {i: [] for i in range(6)}
    
    label_dir = os.path.join(DATASET_DIR, "train", "labels")
    if os.path.exists(label_dir):
        for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        w = float(parts[3]) * 200  # Piksel cinsine çevir
                        h = float(parts[4]) * 200
                        widths[class_id].append(w)
                        heights[class_id].append(h)
    
    # Genişlik dağılımı
    for class_id in range(6):
        if widths[class_id]:
            axes[0].hist(widths[class_id], bins=20, alpha=0.6, 
                        color=COLORS[class_id], label=CLASS_NAMES_TR[class_id])
    axes[0].set_title('BBox Genişlik Dağılımı (px)', fontsize=13)
    axes[0].set_xlabel('Genişlik (piksel)')
    axes[0].set_ylabel('Frekans')
    axes[0].legend(fontsize=9)
    
    # Yükseklik dağılımı
    for class_id in range(6):
        if heights[class_id]:
            axes[1].hist(heights[class_id], bins=20, alpha=0.6, 
                        color=COLORS[class_id], label=CLASS_NAMES_TR[class_id])
    axes[1].set_title('BBox Yükseklik Dağılımı (px)', fontsize=13)
    axes[1].set_xlabel('Yükseklik (piksel)')
    axes[1].set_ylabel('Frekans')
    axes[1].legend(fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(SCREENSHOT_DIR, "03_bbox_analizi.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ BBox analizi kaydedildi: {save_path}")


def plot_dataset_summary():
    """Veri seti özet infografiği."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Başlık
    ax.text(5, 9.5, 'NEU Surface Defect Database', fontsize=24, fontweight='bold',
            ha='center', va='center', color='#2C3E50')
    ax.text(5, 8.8, 'Çelik Yüzey Hatası Tespit Projesi - Veri Seti Özeti', fontsize=14,
            ha='center', va='center', color='#7F8C8D')
    
    # Kutu bilgileri
    info_boxes = [
        ('📸 Toplam Görsel', '1800', '#3498DB'),
        ('🏋️ Eğitim', '1440 (%80)', '#2ECC71'),
        ('🧪 Doğrulama', '360 (%20)', '#E74C3C'),
        ('📐 Boyut', '200×200 px', '#9B59B6'),
        ('🎯 Sınıf Sayısı', '6', '#F39C12'),
        ('📝 Format', 'YOLO TXT', '#1ABC9C'),
    ]
    
    for i, (label, value, color) in enumerate(info_boxes):
        row = i // 3
        col = i % 3
        x = 1.5 + col * 3
        y = 6.5 - row * 2
        
        rect = mpatches.FancyBboxPatch((x - 1.2, y - 0.7), 2.4, 1.4,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.15,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.15, label, fontsize=10, ha='center', va='center', color='#2C3E50')
        ax.text(x, y - 0.25, value, fontsize=16, ha='center', va='center', 
                fontweight='bold', color=color)
    
    # Alt not
    ax.text(5, 2.5, 'Sınıflar: Çatlama | Kapanma | Yamalar | Çukurlu Yüzey | Hadde Tufali | Çizikler',
            fontsize=11, ha='center', va='center', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', edgecolor='#BDC3C7'))
    
    ax.text(5, 1.5, 'Model: YOLOv9c | Framework: Ultralytics | Hedef: %95 Doğruluk',
            fontsize=11, ha='center', va='center', color='#7F8C8D')
    
    save_path = os.path.join(SCREENSHOT_DIR, "00_veri_seti_ozet.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Veri seti özeti kaydedildi: {save_path}")


def plot_annotation_examples():
    """Detaylı annotation görselleri - her sınıftan 2'şer adet."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('NEU-DET - Annotation Örnekleri (Bounding Box ile)', 
                 fontsize=20, fontweight='bold')
    
    train_img_dir = os.path.join(DATASET_DIR, "train", "images")
    train_lbl_dir = os.path.join(DATASET_DIR, "train", "labels")
    
    plot_idx = 0
    for class_id in range(6):
        class_name = CLASS_NAMES[class_id]
        
        # Bu sınıfa ait 2 farklı görsel bul
        for sample_num in [1, 5]:
            row = plot_idx // 4
            col = plot_idx % 4
            ax = axes[row][col]
            
            img_path = os.path.join(train_img_dir, f"{class_name}_{sample_num}.jpg")
            label_path = os.path.join(train_lbl_dir, f"{class_name}_{sample_num}.txt")
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img, cmap='gray')
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        obj_count = 0
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                _, xc, yc, w, h = [float(p) for p in parts]
                                img_w, img_h = img.size
                                x1 = (xc - w/2) * img_w
                                y1 = (yc - h/2) * img_h
                                bw = w * img_w
                                bh = h * img_h
                                rect = mpatches.Rectangle(
                                    (x1, y1), bw, bh,
                                    linewidth=2.5, 
                                    edgecolor=COLORS[class_id],
                                    facecolor=COLORS[class_id],
                                    alpha=0.2
                                )
                                ax.add_patch(rect)
                                rect_border = mpatches.Rectangle(
                                    (x1, y1), bw, bh,
                                    linewidth=2.5, 
                                    edgecolor=COLORS[class_id],
                                    facecolor='none'
                                )
                                ax.add_patch(rect_border)
                                obj_count += 1
                    
                    ax.set_title(f'{CLASS_NAMES_TR[class_id]} #{sample_num}\n({obj_count} nesne)', 
                                fontsize=11, fontweight='bold')
            
            ax.axis('off')
            plot_idx += 1
    
    # Kalan boş hücreleri gizle
    for i in range(plot_idx, 12):
        row = i // 4
        col = i % 4
        axes[row][col].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(SCREENSHOT_DIR, "04_annotation_ornekleri.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Annotation örnekleri kaydedildi: {save_path}")


def main():
    print("=" * 60)
    print("  NEU-DET Veri Seti Görselleştirme")
    print("  Staj Defteri İçin Ekran Görüntüleri")
    print("=" * 60)
    
    print(f"\n  Kayıt dizini: {SCREENSHOT_DIR}\n")
    
    plot_dataset_summary()
    plot_class_distribution()
    plot_sample_images()
    plot_bbox_analysis()
    plot_annotation_examples()
    
    print(f"\n  🎉 Tüm görselleştirmeler tamamlandı!")
    print(f"  📁 Görseller: {SCREENSHOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
