🇬🇧 **English** · [🇹🇷 Türkçe](README.tr.md)

# Steel Surface Defect Detection

YOLOv9 object detection for six classes of surface defect on hot-rolled steel
strip, trained by transfer learning on the public **NEU Surface Defect
Database**.

Built during a summer internship at an integrated iron and steel plant, as a
study project alongside the main tracking work.

> This project uses a **public academic dataset** only — no plant footage is
> involved, so nothing here is withheld.

---

## Dataset

| | |
|---|---|
| Source | NEU Surface Defect Database |
| Images | 1,800 (1,440 train / 360 validation) |
| Resolution | 200×200 px, greyscale (upscaled to 640 for the model) |
| Classes | 6 |

![Dataset samples](docs/dataset-samples.png)

| Class | Turkish | Description |
|-------|---------|-------------|
| `crazing` | Çatlama | Fine network cracking across the surface |
| `inclusion` | Kapanma | Non-metallic inclusions, foreign matter |
| `patches` | Yamalar | Irregular discolouration |
| `pitted_surface` | Çukurlu yüzey | Small pits in the surface |
| `rolled-in_scale` | Hadde tufali | Oxide layer rolled into the surface |
| `scratches` | Çizikler | Linear scratch marks |

Annotations arrive as PASCAL VOC XML; `convert_voc_to_yolo.py` converts
absolute pixel boxes to normalised YOLO coordinates. The conversion was
verified by drawing the converted boxes back onto random images.

## Training

YOLOv9c initialised from COCO-pretrained weights — transfer learning is what
makes 1,800 images enough. Source images are 200×200 but the input is scaled to
640 so the pretrained weights see the scale they expect.

![Training curves](docs/training-curves.png)

## Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.702** |
| mAP@0.5:0.95 | 0.367 |
| Precision | 0.661 |
| Recall | 0.663 |
| F1 | 0.662 |

Per class:

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| patches | 0.741 | 0.891 | **0.899** |
| inclusion | 0.715 | 0.774 | 0.836 |
| pitted_surface | 0.847 | 0.690 | 0.797 |
| scratches | 0.517 | 0.868 | 0.783 |
| crazing | 0.675 | 0.377 | 0.503 |
| rolled-in_scale | 0.471 | 0.379 | **0.395** |

![Confusion matrix](docs/confusion-matrix.png)

The weak classes are the honest story here. `crazing` and `rolled-in_scale`
look alike — fine surface texture at low contrast — and the confusion matrix
shows them being traded for one another. Both would benefit from higher-contrast
imaging rather than more model capacity.

![Predictions](docs/predictions.png)

## Repository layout

| File | Purpose |
|------|---------|
| `convert_voc_to_yolo.py` | PASCAL VOC XML → YOLO label format |
| `visualize_dataset.py` | Class distribution, bbox size analysis, annotation overlays |
| `train.py` | Training entry point |
| `evaluate.py` | Validation metrics, confusion matrix, PR/F1 curves |
| `predict.py` | Inference on new images |
| `app.py` | Gradio demo interface |

## Running it

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
python convert_voc_to_yolo.py
python train.py
python evaluate.py
python app.py          # Gradio demo
```

Source comments are in Turkish; documentation is in English.
