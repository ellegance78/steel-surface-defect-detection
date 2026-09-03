"""
NEU-DET Veri Seti: Pascal VOC XML → YOLO TXT Format Dönüştürücü
================================================================
Bu script, NEU Surface Defect Database'deki Pascal VOC formatındaki
annotation dosyalarını (XML) YOLOv9 eğitimi için gerekli YOLO formatına
(.txt) dönüştürür.

Giriş: Pascal VOC XML → <xmin> <ymin> <xmax> <ymax> (piksel)
Çıkış: YOLO TXT → <class_id> <x_center> <y_center> <width> <height> (normalize 0-1)

Yazar: Berkay
Tarih: 2024
"""

import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================
# KONFİGÜRASYON
# ============================================================

# Kaynak veri seti yolu (masaüstündeki orijinal veri)
SOURCE_DIR = os.path.expanduser("~/Desktop/STAJ/Proje-1/NEU-DET")

# Hedef dizin (proje içinde YOLO formatında)
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "NEU-DET")

# Sınıf eşleştirmesi (class_name → class_id)
CLASS_MAP = {
    "crazing": 0,        # Çatlama
    "inclusion": 1,      # Kapanma/Kalıntı
    "patches": 2,        # Yamalar
    "pitted_surface": 3, # Çukurlu Yüzey
    "rolled-in_scale": 4,# Hadde Tufali
    "scratches": 5       # Çizikler
}

# Veri seti bölümleri
SPLITS = {
    "train": "train",
    "validation": "val"  # Kaynak: validation → Hedef: val
}


def convert_voc_to_yolo(xml_path, img_width, img_height):
    """
    Tek bir VOC XML dosyasını parse eder ve YOLO formatında satırlar döndürür.
    
    YOLO formatı: <class_id> <x_center> <y_center> <width> <height>
    Tüm değerler [0, 1] aralığında normalize edilir.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_lines = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        
        if class_name not in CLASS_MAP:
            print(f"  ⚠ Bilinmeyen sınıf: '{class_name}' → {xml_path}")
            continue
        
        class_id = CLASS_MAP[class_name]
        
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # YOLO formatına dönüştürme (normalize)
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Sınır kontrolü [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines


def process_split(source_split_name, target_split_name):
    """
    Bir veri bölümünü (train/val) işler:
    1. Görselleri alt klasörlerden düz yapıya kopyalar
    2. XML annotation'ları YOLO formatına dönüştürür
    """
    source_img_dir = os.path.join(SOURCE_DIR, source_split_name, "images")
    source_ann_dir = os.path.join(SOURCE_DIR, source_split_name, "annotations")
    
    target_img_dir = os.path.join(TARGET_DIR, target_split_name, "images")
    target_lbl_dir = os.path.join(TARGET_DIR, target_split_name, "labels")
    
    # Hedef dizinleri oluştur
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)
    
    stats = {
        "total_images": 0,
        "total_labels": 0,
        "total_objects": 0,
        "class_counts": {name: 0 for name in CLASS_MAP},
        "skipped_no_xml": [],
        "skipped_no_img": []
    }
    
    print(f"\n{'='*60}")
    print(f"  {source_split_name.upper()} → {target_split_name.upper()} işleniyor...")
    print(f"{'='*60}")
    
    # 1. Tüm görselleri topla (alt klasörlerden)
    all_images = {}
    for class_dir in sorted(os.listdir(source_img_dir)):
        class_path = os.path.join(source_img_dir, class_dir)
        if not os.path.isdir(class_path) or class_dir.startswith('.'):
            continue
        for img_file in sorted(os.listdir(class_path)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images[img_file] = os.path.join(class_path, img_file)
    
    # 2. Tüm XML dosyalarını topla
    all_xmls = {}
    for xml_file in sorted(os.listdir(source_ann_dir)):
        if xml_file.endswith('.xml'):
            base_name = os.path.splitext(xml_file)[0]
            all_xmls[base_name] = os.path.join(source_ann_dir, xml_file)
    
    # 3. Görselleri işle
    for img_file, img_path in tqdm(all_images.items(), desc=f"  {target_split_name}"):
        base_name = os.path.splitext(img_file)[0]
        
        # Görseli kopyala
        shutil.copy2(img_path, os.path.join(target_img_dir, img_file))
        stats["total_images"] += 1
        
        # Eşleşen XML'i bul
        if base_name in all_xmls:
            xml_path = all_xmls[base_name]
            
            # XML'den boyut bilgisini al
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
            
            # Dönüştür
            yolo_lines = convert_voc_to_yolo(xml_path, img_w, img_h)
            
            if yolo_lines:
                label_path = os.path.join(target_lbl_dir, base_name + ".txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")
                stats["total_labels"] += 1
                stats["total_objects"] += len(yolo_lines)
                
                # Sınıf sayılarını güncelle
                for line in yolo_lines:
                    class_id = int(line.split()[0])
                    class_name = [k for k, v in CLASS_MAP.items() if v == class_id][0]
                    stats["class_counts"][class_name] += 1
        else:
            stats["skipped_no_xml"].append(img_file)
    
    # XML'i olan ama görseli olmayan dosyalar
    for base_name in all_xmls:
        found = False
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if base_name + ext in all_images:
                found = True
                break
        if not found:
            stats["skipped_no_img"].append(base_name + ".xml")
    
    return stats


def print_stats(split_name, stats):
    """İstatistikleri formatlanmış şekilde yazdırır."""
    print(f"\n  📊 {split_name.upper()} İstatistikleri:")
    print(f"  ├─ Toplam görsel: {stats['total_images']}")
    print(f"  ├─ Toplam label:  {stats['total_labels']}")
    print(f"  ├─ Toplam nesne:  {stats['total_objects']}")
    print(f"  ├─ Sınıf dağılımı:")
    for class_name, count in stats["class_counts"].items():
        bar = "█" * (count // 10) if count > 0 else ""
        print(f"  │   {class_name:20s}: {count:4d} {bar}")
    
    if stats["skipped_no_xml"]:
        print(f"  ├─ ⚠ XML'siz görsel: {len(stats['skipped_no_xml'])} adet")
        for f in stats["skipped_no_xml"][:5]:
            print(f"  │   → {f}")
    
    if stats["skipped_no_img"]:
        print(f"  └─ ⚠ Görselsiz XML: {len(stats['skipped_no_img'])} adet")
        for f in stats["skipped_no_img"][:5]:
            print(f"  │   → {f}")
    
    print(f"  └─ ✅ Tamamlandı!")


def create_data_yaml():
    """YOLO eğitimi için data.yaml konfigürasyon dosyasını oluşturur."""
    yaml_content = f"""# NEU Surface Defect Database - YOLOv9 Konfigürasyonu
# Çelik Yüzey Hatası Tespit Projesi
# ====================================================

path: {os.path.abspath(TARGET_DIR)}
train: train/images
val: val/images

# Sınıf sayısı
nc: {len(CLASS_MAP)}

# Sınıf isimleri
names:
  0: crazing          # Çatlama
  1: inclusion        # Kapanma/Kalıntı
  2: patches          # Yamalar
  3: pitted_surface   # Çukurlu Yüzey
  4: rolled-in_scale  # Hadde Tufali
  5: scratches        # Çizikler
"""
    
    yaml_path = os.path.join(TARGET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"\n  📄 data.yaml oluşturuldu: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("  NEU-DET Veri Seti Dönüştürücü")
    print("  Pascal VOC XML → YOLO TXT Format")
    print("=" * 60)
    print(f"\n  Kaynak: {SOURCE_DIR}")
    print(f"  Hedef:  {TARGET_DIR}")
    print(f"  Sınıflar: {list(CLASS_MAP.keys())}")
    
    # Her bölümü işle
    all_stats = {}
    for source_name, target_name in SPLITS.items():
        stats = process_split(source_name, target_name)
        all_stats[target_name] = stats
        print_stats(target_name, stats)
    
    # data.yaml oluştur
    create_data_yaml()
    
    # Genel özet
    print(f"\n{'='*60}")
    print(f"  GENEL ÖZET")
    print(f"{'='*60}")
    total_imgs = sum(s["total_images"] for s in all_stats.values())
    total_objs = sum(s["total_objects"] for s in all_stats.values())
    print(f"  Toplam görsel: {total_imgs}")
    print(f"  Toplam nesne:  {total_objs}")
    print(f"  Hedef dizin:   {TARGET_DIR}")
    print(f"\n  ✅ Dönüştürme başarıyla tamamlandı!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
