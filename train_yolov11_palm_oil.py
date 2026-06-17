"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         YOLOv11 Training Pipeline — Palm Cooking Oil Detection              ║
║         Dataset: palm-cooking-oil-lfnhd (Roboflow Universe)                ║
║         Goal  : Production-grade, real-world accurate detection             ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE
-----
  # Step 1 — Install dependencies  (run once)
  pip install ultralytics roboflow torch torchvision torchaudio --upgrade

  # Step 2 — Run this script
  python train_yolov11_palm_oil.py

  # Step 3 — Validate / export / infer (see helper scripts at the bottom)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ▸ IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import shutil
import subprocess
import time
import json
from pathlib import Path

# ── Try importing heavy deps; give a clear error if missing ──────────────────
try:
    import torch
    from ultralytics import YOLO
    import yaml
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("Run:  pip install ultralytics torch torchvision --upgrade")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ▸ CONFIGURATION  (edit these values to customise your run)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Roboflow credentials ─────────────────────────────────────────────────
    "ROBOFLOW_API_KEY": "ZkRXb7lxKXo2w2JHwcVy",   # ← paste your Roboflow API key
    "WORKSPACE"       : "smart-traffic-vmglm",    # ← workspace slug from URL
    "PROJECT"         : "palm-cooking-oil-lfnhd",  # ← project slug from URL
    "VERSION"         : 1,                     # ← dataset version number

    # ── Dataset paths ────────────────────────────────────────────────────────
    # If you already downloaded the dataset, set this to that folder.
    # Otherwise leave "" and the script will auto-download via Roboflow API.
    "DATASET_DIR"     : "C:/Users/Robin/Downloads/oil/Palm Cooking Oil.yolov11",   # e.g. "/home/user/datasets/palm-cooking-oil-1"

    # ── Model selection ──────────────────────────────────────────────────────
    # yolo11n  → fastest,  lowest accuracy  (great for edge / Raspberry Pi)
    # yolo11s  → fast,     good accuracy    (good for mobile / embedded)
    # yolo11m  → balanced  ← RECOMMENDED for real-world palm-oil detection
    # yolo11l  → accurate, slower
    # yolo11x  → most accurate, GPU-heavy
    "MODEL_SIZE"      : "yolo11m",

    # ── Training hyperparameters ─────────────────────────────────────────────
    "EPOCHS"          : 150,          # increase to 200-300 for best results
    "IMGSZ"           : 640,          # 640 standard; try 1280 if GPU allows
    "BATCH"           : 16,           # reduce to 8 if you get OOM errors
    "WORKERS"         : 8,            # CPU data-loader threads
    "PATIENCE"        : 30,           # early-stop if mAP doesn't improve

    # ── Optimizer ────────────────────────────────────────────────────────────
    "OPTIMIZER"       : "AdamW",      # AdamW or SGD
    "LR0"             : 0.001,        # initial learning rate
    "LRF"             : 0.01,         # final LR = LR0 * LRF
    "MOMENTUM"        : 0.937,
    "WEIGHT_DECAY"    : 0.0005,
    "WARMUP_EPOCHS"   : 5,

    # ── Augmentation (tuned for cooking oil bottle detection) ────────────────
    "HSV_H"           : 0.015,        # hue jitter        (label colour shift)
    "HSV_S"           : 0.7,          # saturation jitter (oily-shiny surfaces)
    "HSV_V"           : 0.4,          # brightness jitter (shelf / studio light)
    "DEGREES"         : 10,           # rotation (bottles tilt slightly)
    "TRANSLATE"       : 0.1,
    "SCALE"           : 0.5,          # scale jitter — handles close/far bottles
    "SHEAR"           : 2.0,
    "PERSPECTIVE"     : 0.0001,
    "FLIPUD"          : 0.0,          # bottles are never upside-down in retail
    "FLIPLR"          : 0.5,          # horizontal flip OK
    "MOSAIC"          : 1.0,          # mosaic augmentation (critical for mAP)
    "MIXUP"           : 0.15,         # mild mixup (good for overlapping items)
    "COPY_PASTE"      : 0.0,

    # ── Output ───────────────────────────────────────────────────────────────
    "PROJECT_NAME"    : "palm_oil_yolov11",
    "RUN_NAME"        : "train_v1",
    "SAVE_DIR"        : "runs/detect",

    # ── Post-training ────────────────────────────────────────────────────────
    "EXPORT_FORMAT"   : "onnx",       # "onnx" | "torchscript" | "tflite" | None
    "CONF_THRESHOLD"  : 0.25,         # detection confidence threshold
    "IOU_THRESHOLD"   : 0.45,         # NMS IoU threshold
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ▸ ENVIRONMENT CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_environment():
    print("\n" + "="*70)
    print("  ENVIRONMENT CHECK")
    print("="*70)

    # GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ GPU  : {gpu}  ({vram:.1f} GB VRAM)")

        # Auto-adjust batch size based on VRAM
        if vram < 6 and CFG["BATCH"] > 8:
            CFG["BATCH"] = 8
            print(f"  ⚠️  Low VRAM — batch auto-reduced to {CFG['BATCH']}")
        if vram < 4 and CFG["IMGSZ"] > 640:
            CFG["IMGSZ"] = 640
            print(f"  ⚠️  Low VRAM — imgsz forced to 640")
    else:
        print("  ⚠️  No GPU detected — training on CPU (very slow!)")
        print("      Tip: Use Google Colab (free T4 GPU) for faster training.")
        CFG["BATCH"] = 4
        CFG["WORKERS"] = 2

    # Python & package versions
    import ultralytics
    print(f"  ✅ Python     : {sys.version.split()[0]}")
    print(f"  ✅ PyTorch    : {torch.__version__}")
    print(f"  ✅ Ultralytics: {ultralytics.__version__}")
    print(f"  ✅ CUDA       : {torch.version.cuda or 'N/A'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ▸ DATASET DOWNLOAD (Roboflow)
# ─────────────────────────────────────────────────────────────────────────────
def download_dataset() -> Path:
    """
    Download the palm-cooking-oil dataset from Roboflow in YOLOv8 format
    (fully compatible with YOLOv11 / Ultralytics 8.x).
    Returns the path to the dataset directory.
    """
    # If user already has dataset on disk, skip download
    if CFG["DATASET_DIR"]:
        dataset_path = Path(CFG["DATASET_DIR"])
        if dataset_path.exists() and (dataset_path / "data.yaml").exists():
            print(f"  ✅ Using existing dataset: {dataset_path}")
            return dataset_path
        else:
            print(f"  ⚠️  DATASET_DIR set but path invalid: {dataset_path}")

    print("\n" + "="*70)
    print("  DOWNLOADING DATASET FROM ROBOFLOW")
    print("="*70)

    if CFG["ROBOFLOW_API_KEY"] == "YOUR_API_KEY_HERE":
        print("""
  [ERROR] No Roboflow API key set!

  How to get your API key:
    1. Go to https://roboflow.com and log in
    2. Click your avatar → Settings → Roboflow API
    3. Copy your Private API Key
    4. Paste it into CFG['ROBOFLOW_API_KEY'] at the top of this script
       OR export it as an environment variable:
           export ROBOFLOW_API_KEY="your_key_here"

  Alternatively, manually download the dataset:
    1. Go to: https://universe.roboflow.com/palm-cooking-oil/palm-cooking-oil-lfnhd/1
    2. Click Export → YOLOv8 format → Download ZIP
    3. Unzip to a folder and set CFG['DATASET_DIR'] = 'path/to/that/folder'
""")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])
        from roboflow import Roboflow

    api_key = CFG["ROBOFLOW_API_KEY"] or os.environ.get("ROBOFLOW_API_KEY", "")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(CFG["WORKSPACE"]).project(CFG["PROJECT"])
    version = project.version(CFG["VERSION"])
    dataset = version.download("yolov8")   # yolov8 format = compatible with YOLO11

    dataset_path = Path(dataset.location)
    print(f"\n  ✅ Dataset downloaded to: {dataset_path}")
    return dataset_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ▸ DATA YAML VALIDATION & AUTO-FIX
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_fix_yaml(dataset_path: Path) -> Path:
    """
    Roboflow's exported data.yaml sometimes uses relative paths that confuse
    Ultralytics when the script is run from a different working directory.
    This function rewrites all paths to absolute paths and validates the file.
    """
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    print("\n" + "="*70)
    print("  DATA.YAML VALIDATION")
    print("="*70)
    print(f"  Classes ({data.get('nc', '?')}): {data.get('names', [])}")

    # ── Rewrite paths to absolute ────────────────────────────────────────────
    for split in ("train", "val", "test"):
        if split in data:
            p = Path(data[split])
            if not p.is_absolute():
                abs_p = (dataset_path / p).resolve()
                if abs_p.exists():
                    data[split] = str(abs_p)
                    print(f"  ✅ {split:5s}: {abs_p}")
                else:
                    # Common Roboflow layout: train/images, valid/images
                    for candidate in [
                        dataset_path / split / "images",
                        dataset_path / "valid" / "images",
                        dataset_path / split,
                    ]:
                        if candidate.exists():
                            data[split] = str(candidate.resolve())
                            print(f"  ✅ {split:5s}: {candidate.resolve()}")
                            break

    # Remove Roboflow-specific fields that Ultralytics doesn't need
    for key in ("roboflow", "download"):
        data.pop(key, None)

    fixed_yaml = dataset_path / "data_fixed.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"\n  ✅ Fixed YAML saved → {fixed_yaml}")
    return fixed_yaml


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ▸ TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(data_yaml: Path) -> Path:
    print("\n" + "="*70)
    print("  STARTING YOLOv11 TRAINING")
    print("="*70)
    print(f"  Model   : {CFG['MODEL_SIZE']}.pt")
    print(f"  Epochs  : {CFG['EPOCHS']}")
    print(f"  Imgsz   : {CFG['IMGSZ']}")
    print(f"  Batch   : {CFG['BATCH']}")
    print(f"  Device  : {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    print()

    # Load pretrained backbone (auto-downloads ~50 MB on first run)
    model = YOLO(f"{CFG['MODEL_SIZE']}.pt")

    # ── Full hyperparameter training call ────────────────────────────────────
    results = model.train(
        data        = str(data_yaml),
        epochs      = CFG["EPOCHS"],
        imgsz       = CFG["IMGSZ"],
        batch       = CFG["BATCH"],
        workers     = CFG["WORKERS"],
        patience    = CFG["PATIENCE"],

        # Optimiser
        optimizer   = CFG["OPTIMIZER"],
        lr0         = CFG["LR0"],
        lrf         = CFG["LRF"],
        momentum    = CFG["MOMENTUM"],
        weight_decay= CFG["WEIGHT_DECAY"],
        warmup_epochs = CFG["WARMUP_EPOCHS"],

        # Augmentation
        hsv_h       = CFG["HSV_H"],
        hsv_s       = CFG["HSV_S"],
        hsv_v       = CFG["HSV_V"],
        degrees     = CFG["DEGREES"],
        translate   = CFG["TRANSLATE"],
        scale       = CFG["SCALE"],
        shear       = CFG["SHEAR"],
        perspective = CFG["PERSPECTIVE"],
        flipud      = CFG["FLIPUD"],
        fliplr      = CFG["FLIPLR"],
        mosaic      = CFG["MOSAIC"],
        mixup       = CFG["MIXUP"],
        copy_paste  = CFG["COPY_PASTE"],

        # Detection thresholds
        conf        = CFG["CONF_THRESHOLD"],
        iou         = CFG["IOU_THRESHOLD"],

        # Output
        project     = CFG["PROJECT_NAME"],
        name        = CFG["RUN_NAME"],
        save        = True,
        save_period = 10,             # checkpoint every 10 epochs
        plots       = True,           # loss/mAP curves, confusion matrix
        verbose     = True,

        # Reproducibility
        seed        = 42,
        deterministic = True,

        # Performance tricks
        amp         = True,           # Automatic Mixed Precision (fp16)
        cache       = "ram",          # cache images in RAM for speed
                                      # use cache="disk" if RAM < 16 GB
        close_mosaic = 10,            # disable mosaic for last 10 epochs
        label_smoothing = 0.05,       # regularisation — helps generalisation
        rect        = False,          # rectangular training (faster, less aug)
        cos_lr      = True,           # cosine LR scheduler
        multi_scale = False,          # enable for better scale robustness
    )

    # Path to the best checkpoint
    best_pt = Path(CFG["PROJECT_NAME"]) / CFG["RUN_NAME"] / "weights" / "best.pt"
    print(f"\n  ✅ Training complete!")
    print(f"  🏆 Best weights : {best_pt.resolve()}")
    return best_pt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ▸ VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate(best_pt: Path, data_yaml: Path):
    print("\n" + "="*70)
    print("  VALIDATION")
    print("="*70)

    model = YOLO(str(best_pt))
    metrics = model.val(
        data    = str(data_yaml),
        imgsz   = CFG["IMGSZ"],
        batch   = CFG["BATCH"],
        conf    = CFG["CONF_THRESHOLD"],
        iou     = CFG["IOU_THRESHOLD"],
        plots   = True,
        verbose = True,
    )

    # Pretty-print key metrics
    print("\n  ── KEY METRICS ──────────────────────────────────────")
    print(f"  mAP@50        : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95     : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print("  ─────────────────────────────────────────────────────")
    print()

    # Save metrics to JSON for later reference
    metrics_out = {
        "map50"    : float(metrics.box.map50),
        "map50_95" : float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall"   : float(metrics.box.mr),
    }
    out_dir = best_pt.parent.parent
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics saved → {out_dir / 'final_metrics.json'}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ▸ EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def export_model(best_pt: Path):
    fmt = CFG.get("EXPORT_FORMAT")
    if not fmt:
        return

    print("\n" + "="*70)
    print(f"  EXPORTING MODEL → {fmt.upper()}")
    print("="*70)

    model = YOLO(str(best_pt))
    export_path = model.export(
        format  = fmt,
        imgsz   = CFG["IMGSZ"],
        dynamic = False,        # static shapes — faster for production
        simplify = True,        # ONNX graph simplification
        opset   = 17,           # ONNX opset (17 = widely supported in 2025)
        half    = False,        # FP32 by default; set True for TensorRT/TFLite
    )
    print(f"\n  ✅ Exported to: {export_path}")
    return export_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ▸ QUICK INFERENCE DEMO
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_demo(best_pt: Path, source: str = "0"):
    """
    source can be:
      "0"         → webcam
      "image.jpg" → single image
      "video.mp4" → video file
      "path/"     → folder of images
    """
    print("\n" + "="*70)
    print("  INFERENCE DEMO")
    print("="*70)
    print(f"  Source: {source}")

    model = YOLO(str(best_pt))
    results = model.predict(
        source      = source,
        imgsz       = CFG["IMGSZ"],
        conf        = CFG["CONF_THRESHOLD"],
        iou         = CFG["IOU_THRESHOLD"],
        save        = True,
        save_txt    = True,
        save_conf   = True,
        show        = False,   # set True to pop up a window
        line_width  = 2,
        project     = CFG["PROJECT_NAME"],
        name        = "inference",
    )

    for r in results:
        n = len(r.boxes)
        print(f"  Detected {n} object(s) in {Path(r.path).name}")

    print(f"\n  Results saved → {CFG['PROJECT_NAME']}/inference/")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ▸ HYPERPARAMETER TUNING  (optional — comment out to skip)
# ─────────────────────────────────────────────────────────────────────────────
def run_hyperparameter_tuning(data_yaml: Path):
    """
    Ultralytics built-in evolutionary hyperparameter search.
    Runs `iterations` experiments and picks the best combination.
    ⚠️  This is slow (each iteration = one training run).  Use only when
        you have time and want to squeeze the last few mAP points.
    """
    print("\n" + "="*70)
    print("  HYPERPARAMETER TUNING  (evolutionary search)")
    print("="*70)

    model = YOLO(f"{CFG['MODEL_SIZE']}.pt")
    model.tune(
        data       = str(data_yaml),
        epochs     = 30,          # short epochs per trial
        iterations = 20,          # number of trials
        optimizer  = "AdamW",
        plots      = True,
        save       = True,
        val        = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ▸ MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║   Palm Cooking Oil — YOLOv11 Training Pipeline                          ║
║   Dataset : palm-cooking-oil-lfnhd  (Roboflow Universe)                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    t_start = time.time()

    # ── 1. Environment check
    check_environment()

    # ── 2. Dataset
    dataset_path = download_dataset()

    # ── 3. Fix YAML
    data_yaml = validate_and_fix_yaml(dataset_path)

    # ── 4. Train
    best_pt = train(data_yaml)

    # ── 5. Validate
    validate(best_pt, data_yaml)

    # ── 6. Export
    export_model(best_pt)

    # ── 7. (Optional) Inference demo on a test image from the dataset
    test_img_dir = dataset_path / "test" / "images"
    if test_img_dir.exists():
        test_imgs = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
        if test_imgs:
            run_inference_demo(best_pt, source=str(test_imgs[0]))

    elapsed = (time.time() - t_start) / 60
    print(f"\n  ⏱  Total time: {elapsed:.1f} min")
    print(f"\n  🎉 All done!  Weights → {best_pt.resolve()}\n")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# ▸ STANDALONE HELPER SCRIPTS
#   Copy-paste these into their own .py files as needed.
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATE_SCRIPT = '''
# validate_best.py  —  Re-validate your best checkpoint
from ultralytics import YOLO
model = YOLO("palm_oil_yolov11/train_v1/weights/best.pt")
metrics = model.val(data="path/to/data_fixed.yaml", imgsz=640, conf=0.25, iou=0.45, plots=True)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
'''

INFER_SCRIPT = '''
# infer.py  —  Run detection on any image / video / webcam
from ultralytics import YOLO
model = YOLO("palm_oil_yolov11/train_v1/weights/best.pt")

# Image
results = model.predict("test.jpg", conf=0.25, iou=0.45, save=True, show=True)

# Webcam (live)
# results = model.predict(source=0, conf=0.25, show=True, stream=True)

# Video
# results = model.predict("video.mp4", conf=0.25, save=True)
'''

EXPORT_SCRIPT = '''
# export_onnx.py  —  Export to ONNX for production deployment
from ultralytics import YOLO
model = YOLO("palm_oil_yolov11/train_v1/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=17)
# For TensorRT (NVIDIA GPU production):
# model.export(format="engine", imgsz=640, half=True)
# For TFLite (mobile / edge):
# model.export(format="tflite", imgsz=640, int8=True)
'''

TRACK_SCRIPT = '''
# track.py  —  Object tracking in video (ByteTrack)
from ultralytics import YOLO
model = YOLO("palm_oil_yolov11/train_v1/weights/best.pt")
results = model.track(
    source="factory_line.mp4",
    conf=0.25, iou=0.45,
    tracker="bytetrack.yaml",
    persist=True,
    save=True,
    show=True,
)
'''
