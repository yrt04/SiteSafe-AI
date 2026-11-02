# Milestone 3 — Held-Out Test Split Evaluation Report (`141` Test Images)

- **Evaluated Model:** `runs/detect/train/weights/best.pt` (YOLOv11n, Epoch 49 best checkpoint)
- **Dataset Split:** `test` (`141` unseen images from `datasets/construction-ppe/images/test`)
- **Hardware:** `NVIDIA GeForce RTX 3050 Laptop GPU` (`imgsz=640`, `batch=16`)
- **Test Inference Latency:** `8.26 ms` inference (`15.84 ms` end-to-end / `63.1 FPS` theoretical throughput)

## 1. Per-Class Test Split Performance (`best.pt` on 141 Test Images)

| Class ID | Class Name | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 | Group |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `6` | **Person** *(Core)* | **82.3%** (`0.8226`) | **83.1%** (`0.8305`) | **84.5%** (`0.8449`) | **52.1%** (`0.5206`) | Core PPE |
| `0` | **Helmet (`helmet`)** *(Core)* | **91.3%** (`0.9129`) | **91.1%** (`0.9115`) | **93.1%** (`0.9313`) | **50.2%** (`0.5017`) | Core PPE |
| `7` | **No-Helmet (`no_helmet`)** *(Core)* | **48.0%** (`0.4798`) | **32.5%** (`0.3250`) | **30.4%** (`0.3041`) | **9.7%** (`0.0969`) | Core PPE |
| `2` | **Vest (`vest`)** *(Core)* | **80.7%** (`0.8070`) | **87.1%** (`0.8708`) | **89.0%** (`0.8901`) | **55.0%** (`0.5504`) | Core PPE |
| `1` | `gloves` | 78.5% (`0.7847`) | 76.0% (`0.7604`) | 78.7% (`0.7872`) | 39.1% (`0.3911`) | Secondary / Fine-Grained |
| `3` | `boots` | 73.3% (`0.7327`) | 71.1% (`0.7109`) | 74.1% (`0.7405`) | 39.2% (`0.3916`) | Secondary / Fine-Grained |
| `4` | `goggles` | 73.4% (`0.7344`) | 80.8% (`0.8077`) | 83.9% (`0.8387`) | 33.0% (`0.3303`) | Secondary / Fine-Grained |
| `5` | `none` | 45.4% (`0.4537`) | 58.8% (`0.5878`) | 47.6% (`0.4756`) | 15.1% (`0.1515`) | Secondary / Fine-Grained |
| `8` | `no_goggle` | 38.4% (`0.3838`) | 17.0% (`0.1704`) | 19.1% (`0.1905`) | 6.3% (`0.0628`) | Secondary / Fine-Grained |
| `9` | `no_gloves` | 29.9% (`0.2994`) | 10.3% (`0.1034`) | 12.1% (`0.1212`) | 2.7% (`0.0272`) | Secondary / Fine-Grained |
| `10` | `no_boots` | 58.3% (`0.5827`) | 4.3% (`0.0435`) | 8.0% (`0.0801`) | 1.0% (`0.0104`) | Secondary / Fine-Grained |
| — | **Core PPE Average (`Person`, `helmet`, `no_helmet`, `vest`)** | **75.6%** (`0.7556`) | **73.4%** (`0.7344`) | **74.3%** (`0.7426`) | **41.7%** (`0.4174`) | **Core 4 Subset** |
| — | Secondary Classes Average (Remaining 7 classes) | 56.7% (`0.5674`) | 45.5% (`0.4549`) | 46.2% (`0.4620`) | 19.5% (`0.1950`) | Non-Core 7 Subset |
| — | **Overall All-Class Average (All 11 Classes)** | **63.6%** (`0.6358`) | **55.7%** (`0.5565`) | **56.4%** (`0.5640`) | **27.6%** (`0.2759`) | **Macro Average** |

## 2. Isolated Core PPE mAP@50 Analysis

- **Overall 11-Class Test mAP@50:** `56.40%` *(Validation split mAP@50 during training was `58.20%` at Epoch 49 / `58.03%` at Epoch 50)*
- **Isolated Core PPE Test mAP@50 (`Person`, `helmet`, `no_helmet`, `vest`):** **`74.26%`** (`+17.86` percentage points vs. 11-class average)
- **Secondary 7-Class Test mAP@50 (`gloves`, `boots`, `goggles`, `none`, `no_goggle`, `no_gloves`, `no_boots`):** `46.20%`
- **Takeaway:** Under-represented, fine-grained negative classes (`no_boots`: `8.0%` on 23 instances, `no_gloves`: `12.1%` on 58 instances, `no_goggle`: `19.1%` on 33 instances, and `none`: `47.6%`) heavily depress the unweighted 11-class macro average (`56.40%` test / `58.20%` val). Isolating the 4 Core PPE classes raises mAP@50 to **`74.26%`** (`+17.86` pp), with `helmet` (`93.1%`), `vest` (`89.0%`), and `Person` (`84.5%`) averaging **`88.88%` mAP@50** (while `no_helmet` at `30.4%` reflects class imbalance with only `40` test instances).

## 3. Model Weights & Architecture Comparison

Workspace weight audit found `3` local `.pt` files: `runs/detect/train/weights/best.pt`, `runs/detect/train/weights/last.pt`, and `yolo11n.pt` (base COCO checkpoint).

| Model / Checkpoint | Params (M) | GFLOPs (@640) | Split / Benchmark | All-Class mAP@50 | Core PPE mAP@50 | All-Class mAP@50-95 | GPU Inference (ms/img) | Notes |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **SiteSafe YOLO11n (`best.pt`, Ep. 49)** | `2.59M` | `6.5` | **PPE `test` (141 imgs)** | **`56.4%`** | **`74.3%`** | **`27.6%`** | `8.26 ms` | Deployed production checkpoint |
| SiteSafe YOLO11n (`best.pt`, Ep. 49) | `2.59M` | `6.5` | PPE `val` (143 imgs) | `58.2%` | — | `29.0%` | — | Best validation fitness checkpoint |
| SiteSafe YOLO11n (`last.pt`, Ep. 50) | `2.59M` | `6.5` | PPE `test` (141 imgs) | `56.1%` | `74.1%` | `27.4%` | `6.61 ms` | Final training epoch checkpoint |
| Pretrained `yolo11n.pt` (Zero-Shot COCO) | `2.62M` | `6.7` | COCO val2017 / PPE `test` | `N/A (0% PPE)` | `N/A (Person only)` | `39.5%` *(COCO)* | `89.60 ms` *(E2E)* | Detects `251` persons on test set; `0` PPE classes without fine-tuning |
| Reference `YOLO11s` (Ultralytics Std.) | `9.41M` | `21.5` | COCO val2017 Ref. | — | — | `47.0%` *(COCO)* | ~`2.5–3.4x` slower | Higher capacity (`3.6x` params, `3.4x` FLOPs); heavier for 30 FPS 4GB Laptop GPU |
| Reference `YOLO11m` (Ultralytics Std.) | `20.09M` | `68.0` | COCO val2017 Ref. | — | — | `51.5%` *(COCO)* | ~`6–8x` slower | Server-grade tier; exceeds lightweight edge budget |

## 4. Saved Report Evidence (`report_evidence/`)

- `report_evidence/confusion_matrix.png`
- `report_evidence/confusion_matrix_normalized.png`
- `report_evidence/test_BoxPR_curve.png`
- `report_evidence/test_BoxF1_curve.png`
- `report_evidence/test_predictions_coco.json`
- `report_evidence/results.png`
- `report_evidence/sample_test_pred_1_image1152.jpg` — Source `image1152.jpg` (`3` boxes, mean conf `73.52%`, max conf `83.12%`, classes: `{'Person': 1, 'no_helmet': 1, 'none': 1}`)
- `report_evidence/sample_test_pred_2_image161.jpg` — Source `image161.jpg` (`3` boxes, mean conf `87.69%`, max conf `91.50%`, classes: `{'vest': 1, 'helmet': 1, 'Person': 1}`)
- `report_evidence/sample_test_pred_3_image550.jpg` — Source `image550.jpg` (`3` boxes, mean conf `87.19%`, max conf `90.43%`, classes: `{'vest': 1, 'helmet': 1, 'Person': 1}`)
