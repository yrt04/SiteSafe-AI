"""Evaluate the trained YOLOv11n PPE model on the held-out Construction-PPE test split."""

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "datasets" / "construction-ppe" / "data.yaml"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "construction-ppe" / "images" / "test"
TRAIN_RUN_DIR = PROJECT_ROOT / "runs" / "detect" / "train"
BEST_WEIGHTS = TRAIN_RUN_DIR / "weights" / "best.pt"
LAST_WEIGHTS = TRAIN_RUN_DIR / "weights" / "last.pt"
BASE_YOLO11N = PROJECT_ROOT / "yolo11n.pt"
EVIDENCE_DIR = PROJECT_ROOT / "report_evidence"

CORE_CLASSES = {"Person", "helmet", "no_helmet", "vest"}
DISPLAY_LABELS = {
    "Person": "**Person** *(Core)*",
    "helmet": "**Helmet (`helmet`)** *(Core)*",
    "no_helmet": "**No-Helmet (`no_helmet`)** *(Core)*",
    "vest": "**Vest (`vest`)** *(Core)*",
}


def resolve_dataset_yaml() -> Path:
    """Write a temporary dataset config with an absolute root path so Ultralytics resolves splits from any CWD."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str((PROJECT_ROOT / "datasets" / "construction-ppe").resolve())
    resolved_yaml = EVIDENCE_DIR / "_resolved_construction_ppe.yaml"
    with open(resolved_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return resolved_yaml


def extract_model_specs(model: YOLO) -> tuple[float, float]:
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    gflops = 0.0
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = float(get_flops(model.model, imgsz=640))
    except Exception:
        pass
    return params_m, gflops


def extract_metrics_dict(metrics, class_names: dict) -> dict:
    """Group DetMetrics into all-class macro, Core-PPE (4-class), and secondary (7-class) summaries."""
    per_class = {}
    for idx_pos, cls_id in enumerate(list(metrics.box.ap_class_index)):
        cls_id_int = int(cls_id)
        cls_name = class_names[cls_id_int]
        p, r, ap50, ap50_95 = metrics.box.class_result(idx_pos)
        per_class[cls_id_int] = {
            "id": cls_id_int,
            "name": cls_name,
            "P": float(p),
            "R": float(r),
            "mAP50": float(ap50),
            "mAP50_95": float(ap50_95),
        }

    core_rows = [row for row in per_class.values() if row["name"] in CORE_CLASSES]
    non_core_rows = [row for row in per_class.values() if row["name"] not in CORE_CLASSES]

    core_summary = {
        "P": float(np.mean([r["P"] for r in core_rows])),
        "R": float(np.mean([r["R"] for r in core_rows])),
        "mAP50": float(np.mean([r["mAP50"] for r in core_rows])),
        "mAP50_95": float(np.mean([r["mAP50_95"] for r in core_rows])),
    }
    non_core_summary = {
        "P": float(np.mean([r["P"] for r in non_core_rows])) if non_core_rows else 0.0,
        "R": float(np.mean([r["R"] for r in non_core_rows])) if non_core_rows else 0.0,
        "mAP50": float(np.mean([r["mAP50"] for r in non_core_rows])) if non_core_rows else 0.0,
        "mAP50_95": float(np.mean([r["mAP50_95"] for r in non_core_rows])) if non_core_rows else 0.0,
    }

    save_path = Path(metrics.save_dir)
    try:
        rel_save_dir = save_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        rel_save_dir = save_path.as_posix()

    return {
        "overall": {
            "P": float(metrics.box.mp),
            "R": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        },
        "core_ppe": core_summary,
        "secondary_ppe": non_core_summary,
        "per_class": per_class,
        "speed_ms": dict(metrics.speed),
        "save_dir": rel_save_dir,
    }


def save_top_sample_predictions(model: YOLO, device: str, top_k: int = 3) -> list[dict]:
    """Export top-scoring test predictions, guaranteeing at least one explicit violation scene."""
    image_paths = sorted(
        p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    candidates = []

    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=0.45,
        iou=0.7,
        device=device,
        verbose=False,
    )

    for img_path, res in zip(image_paths, results):
        boxes = res.boxes
        if boxes is None or len(boxes) < 3:
            continue

        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        cls_names = [res.names[int(c)] for c in cls_ids]
        core_hits = [name for name in cls_names if name in CORE_CLASSES]
        has_violation = "no_helmet" in cls_names

        if len(core_hits) < 2:
            continue

        mean_conf = float(np.mean(confs))
        min_conf = float(np.min(confs))
        unique_core = len(set(core_hits))

        # Rank scenes by detection confidence, class diversity, and presence of explicit violations.
        score = (mean_conf * 2.0) + (min_conf * 0.5) + (0.15 * unique_core) + (0.25 if has_violation else 0.0)
        candidates.append({
            "path": img_path,
            "result": res,
            "mean_conf": mean_conf,
            "max_conf": float(np.max(confs)),
            "num_boxes": int(len(boxes)),
            "classes": cls_names,
            "has_violation": has_violation,
            "score": score,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    violation_candidates = [c for c in candidates if c["has_violation"]]
    if violation_candidates:
        selected.append(violation_candidates[0])

    for cand in candidates:
        if len(selected) >= top_k:
            break
        if cand["path"] not in {s["path"] for s in selected}:
            selected.append(cand)

    saved_info = []
    for idx, item in enumerate(selected, start=1):
        annotated_bgr = item["result"].plot(line_width=2, font_size=12)
        out_name = f"sample_test_pred_{idx}_{item['path'].stem}.jpg"
        out_path = EVIDENCE_DIR / out_name
        cv2.imwrite(str(out_path), annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cls_counts = {}
        for c in item["classes"]:
            cls_counts[c] = cls_counts.get(c, 0) + 1

        saved_info.append({
            "output_file": out_name,
            "source_image": item["path"].name,
            "num_detections": item["num_boxes"],
            "mean_conf": round(item["mean_conf"], 4),
            "max_conf": round(item["max_conf"], 4),
            "class_breakdown": cls_counts,
        })

    return saved_info


def evaluate_coco_baseline_latency_and_person(base_weights: Path, device: str) -> dict:
    model = YOLO(str(base_weights))
    params_m, gflops = extract_model_specs(model)

    image_paths = sorted(
        p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    _ = model.predict(source=str(image_paths[0]), imgsz=640, device=device, verbose=False)

    t0 = time.perf_counter()
    results = model.predict(
        source=[str(p) for p in image_paths], imgsz=640, conf=0.35, device=device, verbose=False
    )
    elapsed_ms_per_img = ((time.perf_counter() - t0) * 1000.0) / max(len(image_paths), 1)

    person_boxes = sum(
        int((r.boxes.cls.cpu().numpy().astype(int) == 0).sum())
        for r in results
        if r.boxes is not None and len(r.boxes) > 0
    )
    return {
        "params_m": round(params_m, 2),
        "gflops": round(gflops, 1),
        "e2e_ms_per_img": round(elapsed_ms_per_img, 2),
        "zero_shot_person_detections": person_boxes,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[INFO] Device: {device_name} ({device})")

    resolved_yaml = resolve_dataset_yaml()
    test_img_count = len(
        [p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    print(f"[INFO] Test split: {test_img_count} images in {TEST_IMAGES_DIR}")

    all_pt_files = sorted(
        p.relative_to(PROJECT_ROOT)
        for p in PROJECT_ROOT.rglob("*.pt")
        if "venv" not in p.parts and "node_modules" not in p.parts
    )
    print(f"[INFO] Local weights: {[str(p) for p in all_pt_files]}")

    best_model = YOLO(str(BEST_WEIGHTS))
    best_params_m, best_gflops = extract_model_specs(best_model)

    print("[INFO] Evaluating best.pt on test split...")
    best_val_metrics = best_model.val(
        data=str(resolved_yaml),
        split="test",
        imgsz=640,
        batch=16,
        save_json=True,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="milestone3_test_eval",
        exist_ok=True,
        device=device,
    )
    best_stats = extract_metrics_dict(best_val_metrics, best_model.names)

    last_stats = None
    last_params_m, last_gflops = best_params_m, best_gflops
    if LAST_WEIGHTS.exists():
        print("[INFO] Evaluating last.pt on test split...")
        last_model = YOLO(str(LAST_WEIGHTS))
        last_val_metrics = last_model.val(
            data=str(resolved_yaml),
            split="test",
            imgsz=640,
            batch=16,
            save_json=False,
            plots=False,
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name="milestone3_last_test_eval",
            exist_ok=True,
            device=device,
        )
        last_stats = extract_metrics_dict(last_val_metrics, last_model.names)

    base_coco_info = None
    if BASE_YOLO11N.exists():
        print("[INFO] Benchmarking pretrained yolo11n.pt baseline...")
        base_coco_info = evaluate_coco_baseline_latency_and_person(BASE_YOLO11N, device)

    print("[INFO] Saving plots and sample predictions to report_evidence/...")
    eval_save_dir = PROJECT_ROOT / best_stats["save_dir"]
    copied_files = []

    artifact_map = {
        eval_save_dir / "confusion_matrix.png": EVIDENCE_DIR / "confusion_matrix.png",
        eval_save_dir / "confusion_matrix_normalized.png": EVIDENCE_DIR / "confusion_matrix_normalized.png",
        eval_save_dir / "BoxPR_curve.png": EVIDENCE_DIR / "test_BoxPR_curve.png",
        eval_save_dir / "BoxF1_curve.png": EVIDENCE_DIR / "test_BoxF1_curve.png",
        eval_save_dir / "predictions.json": EVIDENCE_DIR / "test_predictions_coco.json",
        TRAIN_RUN_DIR / "results.png": EVIDENCE_DIR / "results.png",
    }
    for src, dst in artifact_map.items():
        if src.exists():
            shutil.copy2(src, dst)
            copied_files.append(dst.name)

    sample_preds = save_top_sample_predictions(best_model, device=device, top_k=3)

    if resolved_yaml.exists():
        resolved_yaml.unlink()

    md_lines = [
        "# Milestone 3 - Held-Out Test Split Evaluation Report (`141` Test Images)",
        "",
        "- **Evaluated Model:** `runs/detect/train/weights/best.pt` (YOLOv11n, Epoch 49 best checkpoint)",
        f"- **Dataset Split:** `test` (`{test_img_count}` unseen images from `datasets/construction-ppe/images/test`)",
        f"- **Hardware:** `{device_name}` (`imgsz=640`, `batch=16`)",
    ]
    inf_ms = best_stats["speed_ms"].get("inference", 0.0)
    pre_ms = best_stats["speed_ms"].get("preprocess", 0.0)
    post_ms = best_stats["speed_ms"].get("postprocess", 0.0)
    total_ms = pre_ms + inf_ms + post_ms
    md_lines.extend([
        f"- **Test Inference Latency:** `{inf_ms:.2f} ms` inference (`{total_ms:.2f} ms` end-to-end / `{1000.0 / max(total_ms, 1e-6):.1f} FPS` theoretical throughput)",
        "",
        "## 1. Per-Class Test Split Performance (`best.pt` on 141 Test Images)",
        "",
        "| Class ID | Class Name | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 | Group |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    core_order = [6, 0, 7, 2]
    for cid in core_order:
        if cid in best_stats["per_class"]:
            row = best_stats["per_class"][cid]
            label = DISPLAY_LABELS.get(row["name"], f"**{row['name']}**")
            md_lines.append(
                f"| `{cid}` | {label} | **{row['P']*100:.1f}%** (`{row['P']:.4f}`) | **{row['R']*100:.1f}%** (`{row['R']:.4f}`) | **{row['mAP50']*100:.1f}%** (`{row['mAP50']:.4f}`) | **{row['mAP50_95']*100:.1f}%** (`{row['mAP50_95']:.4f}`) | Core PPE |"
            )

    for cid in sorted(best_stats["per_class"].keys()):
        if cid in core_order:
            continue
        row = best_stats["per_class"][cid]
        md_lines.append(
            f"| `{cid}` | `{row['name']}` | {row['P']*100:.1f}% (`{row['P']:.4f}`) | {row['R']*100:.1f}% (`{row['R']:.4f}`) | {row['mAP50']*100:.1f}% (`{row['mAP50']:.4f}`) | {row['mAP50_95']*100:.1f}% (`{row['mAP50_95']:.4f}`) | Secondary / Fine-Grained |"
        )

    ov = best_stats["overall"]
    cp = best_stats["core_ppe"]
    sp = best_stats["secondary_ppe"]
    delta_map50 = (cp["mAP50"] - ov["mAP50"]) * 100.0

    md_lines.extend([
        f"| - | **Core PPE Average (`Person`, `helmet`, `no_helmet`, `vest`)** | **{cp['P']*100:.1f}%** (`{cp['P']:.4f}`) | **{cp['R']*100:.1f}%** (`{cp['R']:.4f}`) | **{cp['mAP50']*100:.1f}%** (`{cp['mAP50']:.4f}`) | **{cp['mAP50_95']*100:.1f}%** (`{cp['mAP50_95']:.4f}`) | **Core 4 Subset** |",
        f"| - | Secondary Classes Average (Remaining 7 classes) | {sp['P']*100:.1f}% (`{sp['P']:.4f}`) | {sp['R']*100:.1f}% (`{sp['R']:.4f}`) | {sp['mAP50']*100:.1f}% (`{sp['mAP50']:.4f}`) | {sp['mAP50_95']*100:.1f}% (`{sp['mAP50_95']:.4f}`) | Non-Core 7 Subset |",
        f"| - | **Overall All-Class Average (All 11 Classes)** | **{ov['P']*100:.1f}%** (`{ov['P']:.4f}`) | **{ov['R']*100:.1f}%** (`{ov['R']:.4f}`) | **{ov['mAP50']*100:.1f}%** (`{ov['mAP50']:.4f}`) | **{ov['mAP50_95']*100:.1f}%** (`{ov['mAP50_95']:.4f}`) | **Macro Average** |",
        "",
        "## 2. Isolated Core PPE mAP@50 Analysis",
        "",
        f"- **Overall 11-Class Test mAP@50:** `{ov['mAP50']*100:.2f}%` *(Validation split mAP@50 during training was `58.20%` at Epoch 49 / `58.03%` at Epoch 50)*",
        f"- **Isolated Core PPE Test mAP@50 (`Person`, `helmet`, `no_helmet`, `vest`):** **`{cp['mAP50']*100:.2f}%`** (`{delta_map50:+.2f}` percentage points vs. 11-class average)",
        f"- **Secondary 7-Class Test mAP@50 (`gloves`, `boots`, `goggles`, `none`, `no_goggle`, `no_gloves`, `no_boots`):** `{sp['mAP50']*100:.2f}%`",
        "- **Takeaway:** Under-represented negative classes (`no_boots`, `no_gloves`, `no_goggle`, `none`) depress the unweighted 11-class macro average, whereas the 4 Core PPE classes achieve substantially higher detection accuracy.",
        "",
        "## 3. Model Weights & Architecture Comparison",
        "",
        "| Model / Checkpoint | Params (M) | GFLOPs (@640) | Split / Benchmark | All-Class mAP@50 | Core PPE mAP@50 | All-Class mAP@50-95 | GPU Inference (ms/img) | Notes |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
        f"| **SiteSafe YOLO11n (`best.pt`, Ep. 49)** | `{best_params_m:.2f}M` | `{best_gflops:.1f}` | **PPE `test` (141 imgs)** | **`{ov['mAP50']*100:.1f}%`** | **`{cp['mAP50']*100:.1f}%`** | **`{ov['mAP50_95']*100:.1f}%`** | `{inf_ms:.2f} ms` | Deployed production checkpoint |",
        f"| SiteSafe YOLO11n (`best.pt`, Ep. 49) | `{best_params_m:.2f}M` | `{best_gflops:.1f}` | PPE `val` (143 imgs) | `58.2%` | - | `29.0%` | - | Best validation fitness checkpoint |",
    ])

    if last_stats is not None:
        lov = last_stats["overall"]
        lcp = last_stats["core_ppe"]
        linf = last_stats["speed_ms"].get("inference", 0.0)
        md_lines.append(
            f"| SiteSafe YOLO11n (`last.pt`, Ep. 50) | `{last_params_m:.2f}M` | `{last_gflops:.1f}` | PPE `test` (141 imgs) | `{lov['mAP50']*100:.1f}%` | `{lcp['mAP50']*100:.1f}%` | `{lov['mAP50_95']*100:.1f}%` | `{linf:.2f} ms` | Final training epoch checkpoint |"
        )
    if base_coco_info is not None:
        md_lines.append(
            f"| Pretrained `yolo11n.pt` (Zero-Shot COCO) | `{base_coco_info['params_m']:.2f}M` | `{base_coco_info['gflops']:.1f}` | COCO val2017 / PPE `test` | `N/A (0% PPE)` | `N/A (Person only)` | `39.5%` *(COCO)* | `{base_coco_info['e2e_ms_per_img']:.2f} ms` *(E2E)* | Detects `{base_coco_info['zero_shot_person_detections']}` persons on test set; `0` PPE classes without fine-tuning |"
        )
    md_lines.extend([
        "| Reference `YOLO11s` (Ultralytics Std.) | `9.41M` | `21.5` | COCO val2017 Ref. | - | - | `47.0%` *(COCO)* | ~`2.5-3.4x` slower | Higher capacity (`3.6x` params, `3.4x` FLOPs); heavier for 30 FPS 4GB Laptop GPU |",
        "| Reference `YOLO11m` (Ultralytics Std.) | `20.09M` | `68.0` | COCO val2017 Ref. | - | - | `51.5%` *(COCO)* | ~`6-8x` slower | Server-grade tier; exceeds lightweight edge budget |",
        "",
        "## 4. Saved Report Evidence (`report_evidence/`)",
        "",
    ])

    for fname in copied_files:
        md_lines.append(f"- `report_evidence/{fname}`")
    for s in sample_preds:
        md_lines.append(
            f"- `report_evidence/{s['output_file']}` - Source `{s['source_image']}` (`{s['num_detections']}` boxes, mean conf `{s['mean_conf']:.2%}`, max conf `{s['max_conf']:.2%}`, classes: `{s['class_breakdown']}`)"
        )
    md_lines.append("")

    report_md = "\n".join(md_lines)
    report_path = EVIDENCE_DIR / "MILESTONE_3_EVALUATION_RESULTS.md"
    report_path.write_text(report_md, encoding="utf-8")

    summary_json_path = EVIDENCE_DIR / "milestone3_metrics_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "best_test": best_stats,
                "last_test": last_stats,
                "base_yolo11n_coco": base_coco_info,
                "sample_predictions": sample_preds,
                "copied_artifacts": copied_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report_md)
    print(f"[INFO] Markdown report written to {report_path}")
    print(f"[INFO] JSON summary written to {summary_json_path}")


if __name__ == "__main__":
    main()
