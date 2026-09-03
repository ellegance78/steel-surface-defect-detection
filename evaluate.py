"""
YOLOv9 Model Değerlendirme & Raporlama Scripti
================================================
Eğitilmiş modelin performansını analiz eder ve
staj defteri için detaylı raporlar/görseller üretir.
"""

from ultralytics import YOLO
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "runs", "steel_defect_yolov9")
SCREENSHOT_DIR = os.path.join(PROJECT_DIR, "screenshots")
DATA_YAML = os.path.join(PROJECT_DIR, "datasets", "NEU-DET", "data.yaml")

CLASS_NAMES_TR = {
    0: "Çatlama", 1: "Kapanma", 2: "Yamalar",
    3: "Çukurlu Yüzey", 4: "Hadde Tufali", 5: "Çizikler"
}

os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def evaluate_model():
    """Modeli doğrulama seti üzerinde değerlendirir."""
    print("=" * 60)
    print("  YOLOv9 Model Değerlendirme")
    print("=" * 60)
    
    # En iyi modeli yükle
    best_model_path = os.path.join(RESULTS_DIR, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        print(f"  ❌ Model bulunamadı: {best_model_path}")
        print("  ℹ️  Önce train.py ile eğitim yapın.")
        return None
    
    print(f"\n  📂 Model: {best_model_path}")
    model = YOLO(best_model_path)
    
    # Doğrulama
    print("  🔍 Doğrulama yapılıyor...\n")
    results = model.val(data=DATA_YAML, imgsz=640, batch=8, 
                       project=os.path.join(PROJECT_DIR, "runs"),
                       name="steel_defect_eval",
                       exist_ok=True, plots=True)
    
    return results


def generate_report(results):
    """Detaylı performans raporu oluşturur."""
    if results is None:
        return
    
    print("\n" + "=" * 60)
    print("  📊 PERFORMANS RAPORU")
    print("=" * 60)
    
    # Genel metrikler
    metrics = {
        "mAP@0.5": results.box.map50,
        "mAP@0.5:0.95": results.box.map,
        "Precision (Ortalama)": results.box.mp,
        "Recall (Ortalama)": results.box.mr,
    }
    
    print(f"\n  {'Metrik':<25s} {'Değer':>10s}")
    print(f"  {'-'*35}")
    for name, value in metrics.items():
        bar = "█" * int(value * 20)
        print(f"  {name:<25s} {value:>8.4f}  {bar}")
    
    # F1-Score hesaplama
    p = results.box.mp
    r = results.box.mr
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    print(f"  {'F1-Score':<25s} {f1:>8.4f}  {'█' * int(f1 * 20)}")
    
    # Sınıf bazlı metrikler
    print(f"\n\n  {'='*60}")
    print(f"  SINIF BAZLI METRİKLER")
    print(f"  {'='*60}")
    
    print(f"\n  {'Sınıf':<18s} {'Precision':>10s} {'Recall':>10s} {'mAP@0.5':>10s} {'mAP@50:95':>10s}")
    print(f"  {'-'*58}")
    
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    for i, name in enumerate(class_names):
        try:
            ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
            ap = results.box.ap[i] if i < len(results.box.ap) else 0
            p_cls = results.box.p[i] if i < len(results.box.p) else 0
            r_cls = results.box.r[i] if i < len(results.box.r) else 0
            tr_name = CLASS_NAMES_TR.get(i, name)
            print(f"  {tr_name:<18s} {p_cls:>10.4f} {r_cls:>10.4f} {ap50:>10.4f} {ap:>10.4f}")
        except (IndexError, AttributeError):
            pass
    
    print(f"  {'-'*58}")
    print(f"  {'ORTALAMA':<18s} {p:>10.4f} {r:>10.4f} {results.box.map50:>10.4f} {results.box.map:>10.4f}")
    
    # Raporu dosyaya kaydet
    report_path = os.path.join(SCREENSHOT_DIR, "05_performans_raporu.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  YOLOv9 Çelik Yüzey Hatası Tespit - Performans Raporu\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("GENEL METRİKLER:\n")
        for name, value in metrics.items():
            f.write(f"  {name}: {value:.4f}\n")
        f.write(f"  F1-Score: {f1:.4f}\n\n")
        
        f.write("SINIF BAZLI METRİKLER:\n")
        f.write(f"  {'Sınıf':<18s} {'Precision':>10s} {'Recall':>10s} {'mAP@0.5':>10s}\n")
        f.write(f"  {'-'*48}\n")
        for i, name in enumerate(class_names):
            try:
                ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
                p_cls = results.box.p[i] if i < len(results.box.p) else 0
                r_cls = results.box.r[i] if i < len(results.box.r) else 0
                tr_name = CLASS_NAMES_TR.get(i, name)
                f.write(f"  {tr_name:<18s} {p_cls:>10.4f} {r_cls:>10.4f} {ap50:>10.4f}\n")
            except (IndexError, AttributeError):
                pass
    
    print(f"\n  💾 Rapor kaydedildi: {report_path}")


def copy_training_plots():
    """Eğitim sırasında oluşturulan grafikleri screenshots klasörüne kopyalar."""
    import shutil
    
    plots_to_copy = {
        "results.png": "06_egitim_grafikleri.png",
        "confusion_matrix.png": "07_confusion_matrix.png",
        "confusion_matrix_normalized.png": "08_confusion_matrix_normalized.png",
        "BoxF1_curve.png": "09_f1_curve.png",
        "BoxP_curve.png": "10_precision_curve.png",
        "BoxR_curve.png": "11_recall_curve.png",
        "BoxPR_curve.png": "12_precision_recall_curve.png",
        "labels.jpg": "13_label_dagilimi.png",
    }
    
    print(f"\n  📸 Eğitim görselleri kopyalanıyor...")
    copied = 0
    for src_name, dst_name in plots_to_copy.items():
        src = os.path.join(RESULTS_DIR, src_name)
        dst = os.path.join(SCREENSHOT_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ {src_name} → {dst_name}")
            copied += 1
        else:
            print(f"  ⚠️  {src_name} bulunamadı")
    
    # Val sonuçlarındaki görselleri de kopyala
    eval_dir = os.path.join(PROJECT_DIR, "runs", "steel_defect_eval")
    if os.path.exists(eval_dir):
        for src_name, dst_name in plots_to_copy.items():
            src = os.path.join(eval_dir, src_name)
            dst_name_eval = dst_name.replace(".png", "_eval.png")
            dst = os.path.join(SCREENSHOT_DIR, dst_name_eval)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
    
    print(f"\n  📊 Toplam {copied} görsel kopyalandı.")


def main():
    results = evaluate_model()
    generate_report(results)
    copy_training_plots()
    
    print(f"\n{'='*60}")
    print(f"  ✅ Değerlendirme tamamlandı!")
    print(f"  📁 Tüm görseller: {SCREENSHOT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
