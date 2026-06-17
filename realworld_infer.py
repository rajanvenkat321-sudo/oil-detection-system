"""
realworld_infer.py
──────────────────
Production-ready inference for Palm Cooking Oil detection.
Supports: single image, folder batch, live webcam, video file.

Usage:
    # Single image
    python realworld_infer.py --source image.jpg --weights best.pt

    # Folder of images
    python realworld_infer.py --source /path/to/images/ --weights best.pt

    # Live webcam
    python realworld_infer.py --source 0 --weights best.pt --show

    # Video
    python realworld_infer.py --source factory.mp4 --weights best.pt --save
"""

import argparse
import sys
import time
import json
from pathlib import Path

try:
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("Run: pip install ultralytics opencv-python")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "weights"  : "palm_oil_yolov11/train_v1/weights/best.pt",
    "imgsz"    : 640,
    "conf"     : 0.25,     # lower = more detections (more false positives)
    "iou"      : 0.45,     # NMS IoU — higher = less suppression
    "device"   : "cuda:0" if torch.cuda.is_available() else "cpu",
    "save"     : True,
    "save_txt" : False,
    "show"     : False,
    "tracker"  : None,     # "bytetrack.yaml" for tracking in videos
}


def parse_args():
    p = argparse.ArgumentParser(description="Palm Oil YOLOv11 Inference")
    p.add_argument("--weights", default=DEFAULTS["weights"])
    p.add_argument("--source",  default="0",
                   help="Image / folder / video / webcam index")
    p.add_argument("--imgsz",   type=int, default=DEFAULTS["imgsz"])
    p.add_argument("--conf",    type=float, default=DEFAULTS["conf"])
    p.add_argument("--iou",     type=float, default=DEFAULTS["iou"])
    p.add_argument("--device",  default=DEFAULTS["device"])
    p.add_argument("--save",    action="store_true", default=DEFAULTS["save"])
    p.add_argument("--show",    action="store_true", default=DEFAULTS["show"])
    p.add_argument("--track",   action="store_true",
                   help="Enable ByteTrack object tracking (video)")
    return p.parse_args()


def format_detections(results) -> list:
    """Convert YOLO result objects to a clean JSON-serialisable list."""
    output = []
    for r in results:
        frame_dets = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            frame_dets.append({
                "class_id"  : int(box.cls[0]),
                "class_name": r.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox"      : [round(x1, 1), round(y1, 1),
                               round(x2, 1), round(y2, 1)],
                "track_id"  : int(box.id[0]) if box.id is not None else None,
            })
        output.append({
            "source"    : str(r.path),
            "detections": frame_dets,
            "count"     : len(frame_dets),
        })
    return output


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    args = parse_args()

    print("\n" + "="*60)
    print("  Palm Cooking Oil — YOLOv11 Real-World Inference")
    print("="*60)
    print(f"  Weights : {args.weights}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}   IoU: {args.iou}")
    print(f"  Device  : {args.device}")
    print()

    if not Path(args.weights).exists():
        print(f"[ERROR] Weights not found: {args.weights}")
        print("Train first:  python train_yolov11_palm_oil.py")
        sys.exit(1)

    model = YOLO(args.weights)
    device = args.device
    if device.isdigit():
        device = f"cuda:{device}"
    model.to(device)

    t0 = time.time()

    if args.track:
        # ── Video tracking mode ──────────────────────────────────────────────
        results = model.track(
            source      = args.source,
            imgsz       = args.imgsz,
            conf        = args.conf,
            iou         = args.iou,
            tracker     = "bytetrack.yaml",
            persist     = True,
            save        = args.save,
            show        = args.show,
            stream      = True,
            verbose     = False,
        )
        total_frames = 0
        total_detections = 0
        for r in results:
            total_frames += 1
            n = len(r.boxes)
            total_detections += n
            if total_frames % 30 == 0:
                fps = total_frames / (time.time() - t0)
                print(f"  Frame {total_frames:5d} | Objects: {n:3d} | FPS: {fps:.1f}")

        print(f"\n  Total frames   : {total_frames}")
        print(f"  Total detections: {total_detections}")

    else:
        # ── Standard prediction ──────────────────────────────────────────────
        results = model.predict(
            source      = args.source,
            imgsz       = args.imgsz,
            conf        = args.conf,
            iou         = args.iou,
            save        = args.save,
            save_txt    = True,
            save_conf   = True,
            show        = args.show,
            stream      = False,
            verbose     = True,
            augment     = False,     # TTA: set True for slightly better accuracy
        )

        detections = format_detections(results)
        total = sum(d["count"] for d in detections)

        print(f"\n  ── RESULTS ─────────────────────────────────────────")
        for d in detections:
            print(f"  {Path(d['source']).name:40s} → {d['count']} object(s)")
            for det in d["detections"]:
                tid = f" [track {det['track_id']}]" if det["track_id"] else ""
                print(f"      {det['class_name']:25s}  conf={det['confidence']:.2f}  "
                      f"box={det['bbox']}{tid}")

        print(f"\n  Total detections: {total}")
        elapsed = time.time() - t0
        print(f"  Inference time  : {elapsed:.2f}s "
              f"({elapsed/max(len(detections),1)*1000:.1f} ms/image)")

        # Save results as JSON
        out_json = Path("inference_results.json")
        with open(out_json, "w") as f:
            json.dump(detections, f, indent=2)
        print(f"  JSON results    : {out_json.resolve()}")

    print("\n  Done!\n")


if __name__ == "__main__":
    main()
