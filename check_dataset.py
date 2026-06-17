"""
check_dataset.py
────────────────
Run this BEFORE training to verify your downloaded Roboflow dataset is
correctly structured, labels are valid YOLO format, and class balance is good.

Usage:
    python check_dataset.py --dataset /path/to/your/dataset
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv11 dataset health checker")
    p.add_argument("--dataset", type=str, required=True,
                   help="C:/Users/Robin/Downloads/oil/Palm Cooking Oil.yolov11")
    return p.parse_args()


def check_split(split_dir: Path, class_names: list) -> dict:
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    stats = {
        "images"         : 0,
        "labels"         : 0,
        "missing_labels" : [],
        "empty_labels"   : [],
        "class_counts"   : Counter(),
        "corrupt_labels" : [],
    }

    if not img_dir.exists():
        print(f"    ⚠️  No 'images' folder in {split_dir}")
        return stats

    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + \
           list(img_dir.glob("*.jpeg"))
    stats["images"] = len(imgs)

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            stats["missing_labels"].append(img_path.name)
            continue
        stats["labels"] += 1

        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            stats["empty_labels"].append(lbl_path.name)
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                stats["corrupt_labels"].append(lbl_path.name)
                break
            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                # Validate YOLO normalised coords [0, 1]
                if not all(0.0 <= c <= 1.0 for c in coords):
                    stats["corrupt_labels"].append(lbl_path.name)
                    break
                stats["class_counts"][cls_id] += 1
            except ValueError:
                stats["corrupt_labels"].append(lbl_path.name)
                break

    return stats


def print_stats(split_name: str, stats: dict, class_names: list):
    print(f"\n  ── {split_name.upper()} SPLIT ──────────────────────────────────")
    print(f"  Images         : {stats['images']}")
    print(f"  Labels matched : {stats['labels']}")

    if stats["missing_labels"]:
        print(f"  ⚠️  Missing labels: {len(stats['missing_labels'])}")
        for f in stats["missing_labels"][:5]:
            print(f"      {f}")

    if stats["empty_labels"]:
        print(f"  ⚠️  Empty label files: {len(stats['empty_labels'])}")

    if stats["corrupt_labels"]:
        print(f"  ❌ Corrupt labels   : {len(stats['corrupt_labels'])}")
        for f in stats["corrupt_labels"][:5]:
            print(f"      {f}")

    if stats["class_counts"]:
        print(f"\n  Class distribution:")
        total = sum(stats["class_counts"].values())
        for cls_id, count in sorted(stats["class_counts"].items()):
            name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
            bar  = "█" * int(40 * count / max(total, 1))
            print(f"    [{cls_id:2d}] {name:30s} {count:5d}  {bar}")


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    args = parse_args()
    dataset = Path(args.dataset)

    if not dataset.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset}")
        sys.exit(1)

    # Load data.yaml
    yaml_candidates = list(dataset.glob("*.yaml"))
    if not yaml_candidates:
        print(f"[ERROR] No .yaml file found in {dataset}")
        sys.exit(1)

    import yaml
    with open(yaml_candidates[0]) as f:
        cfg = yaml.safe_load(f)

    nc           = cfg.get("nc", 0)
    class_names  = cfg.get("names", [])

    print("\n" + "="*70)
    print("  PALM COOKING OIL — DATASET HEALTH CHECK")
    print("="*70)
    print(f"  Dataset  : {dataset.resolve()}")
    print(f"  Classes  : {nc}  →  {class_names}")

    # Check all three splits
    for split in ("train", "valid", "test"):
        split_dir = dataset / split
        if not split_dir.exists():
            # Try "val" as an alias
            if split == "valid":
                split_dir = dataset / "val"
            if not split_dir.exists():
                print(f"\n  ── {split.upper()} SPLIT ── (not present, skipping)")
                continue

        stats = check_split(split_dir, class_names)
        print_stats(split, stats, class_names)

    # Overall verdict
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)

    if nc < 2:
        print("  ℹ️  Single-class dataset — works fine with YOLO, no changes needed.")
    if nc >= 10:
        print(f"  ℹ️  {nc} classes — consider yolo11m or yolo11l for better accuracy.")

    print("""
  General tips for palm cooking oil detection:
  ─────────────────────────────────────────────────────────────────────
  ✅  Aim for ≥ 200 images per class for robust real-world performance.
  ✅  Ensure images include varied lighting (daylight, fluorescent, LED).
  ✅  Include partial / occluded bottles (common on shelves).
  ✅  Add negative samples (shelves WITHOUT palm oil) to cut false alarms.
  ✅  If mAP < 0.7 after training, add more annotated data.
  ✅  Suggested training target: mAP@50 ≥ 0.85  |  mAP@50-95 ≥ 0.65
  ─────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
