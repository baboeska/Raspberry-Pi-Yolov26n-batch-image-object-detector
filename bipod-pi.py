#!/usr/bin/env python3
"""
Batch-sort images into folders based on YOLO/OpenVINO object detections.
Requires: pip install ultralytics openvino opencv-python tqdm supervision

python3 -m pip install --upgrade \
    ultralytics \
    "ultralytics[export-openvino]" \
    openvino \
    opencv-python \
    pillow \
    tqdm \
    supervision

Example:
    python processor_openvino.py -i ~/Pictures -o ~/Pictures_Sorted \
        -m ./yolo26s_openvino_model/ --mode auto --conf 0.40
"""

import os
import sys
import shutil
import argparse
import csv
import time
import cv2
import supervision as sv
from datetime import datetime
from collections import Counter
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "~/yolo26n_openvino_model"
DEFAULT_SOURCE = "~/Pictures"
DEFAULT_OUTPUT = "~/Picturez"
DEFAULT_CONF = 0.40
DEFAULT_MODE = "auto"
DEFAULT_TILE_THRESHOLD = 1000
DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 100
DEFAULT_MAX_EDGE = 3840

# If multiple classes are detected, the first matching class here wins.
PRIORITY = [
    "person", "car", "bus", "truck",
    "motorcycle", "bicycle", "train", "boat",
    # Livestock and animals
    "cow", "sheep", "horse", "bird", "dog", "cat"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="dsbatch",
        description="Batch-sort images using Ultralytics YOLO with an OpenVINO model.",
    )
    parser.add_argument("--input", "-i", default=DEFAULT_SOURCE, help=f"source directory (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"destination root (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--conf", "-c", type=float, default=DEFAULT_CONF, help=f"confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--model", "-m", default=MODEL_PATH, help=f"path to OpenVINO model (default: {MODEL_PATH})")
    parser.add_argument("--device", default=None, help="optional inference device, e.g. CPU, GPU, or xpu")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size for resize mode (default: 640)")
    parser.add_argument("--dry-run", action="store_true", help="print routing decisions without copying files")
    
    parser.add_argument("--mode", choices=["resize", "tile", "auto"], default=DEFAULT_MODE,
        help="resize: fast; tile: slow, high accuracy; auto: resize small, tile large (default)")
    parser.add_argument("--tile-threshold", type=int, default=DEFAULT_TILE_THRESHOLD,
        help="in auto mode, tile images larger than this (default: 1000)")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help="tile dimensions (default: 640)")
    parser.add_argument("--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP,
        help="tile overlap in pixels (default: 100)")
    parser.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE,
        help="downscale images larger than this before processing (default: 3840 for 4K)")
    return parser.parse_args()


def pick_primary(detected):
    for class_name in PRIORITY:
        if class_name in detected:
            return class_name
    return sorted(detected)[0] if detected else "empty_or_unknown"


def log_failure(log_path, stage, img_path, reason):
    p = Path(log_path)
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if new:
            w.writerow(["timestamp", "stage", "path", "reason"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), stage, str(img_path), reason])


def is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def make_destination(img_path, source_dir, target_folder):
    rel_path = os.path.relpath(img_path, source_dir)
    base_name = os.path.basename(img_path)
    relative_parent = os.path.dirname(rel_path)
    if relative_parent:
        dir_prefix = relative_parent.replace(os.sep, "__")
        stem, extension = os.path.splitext(base_name)
        new_filename = f"{stem}__{dir_prefix}{extension}"
    else:
        new_filename = base_name
    return os.path.join(target_folder, relative_parent, new_filename)


Image.MAX_IMAGE_PIXELS = 100_000_000
def validate_image(path, max_bytes=50 * 1024 * 1024):
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size == 0: return False, "empty or not a file"
        if p.stat().st_size > max_bytes: return False, f"file too large ({p.stat().st_size / 1e6:.1f} MB)"
        with Image.open(p) as im:
            w, h = im.size
            if w <= 0 or h <= 0: return False, "bad dimensions"
            if w * h > Image.MAX_IMAGE_PIXELS: return False, f"decompression bomb ({w}x{h})"
            im.verify()
        return True, "ok"
    except UnidentifiedImageError:
        return False, "not a recognisable image"
    except Exception as e:
        return False, f"unreadable: {e}"


def detect_with_mode(model, image_path, conf_thresh, args):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("cv2.imread returned None")

    long_edge = max(img.shape[:2])

    # Downscale if larger than max_edge (regardless of mode)
    if long_edge > args.max_edge:
        scale = args.max_edge / long_edge
        new_size = (
            max(1, int(img.shape[1] * scale)),
            max(1, int(img.shape[0] * scale))
        )
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        long_edge = max(img.shape[:2])

    # Callback for the slicer. Converts Ultralytics Results -> supervision Detections
    def yolo_predict(arr):
        results = model.predict(
            arr, 
            conf=conf_thresh, 
            imgsz=args.tile_size if (args.mode == "tile" or long_edge >= args.tile_threshold) else args.imgsz,
            device=args.device,
            verbose=False
        )[0]
        
        if results.boxes is None or len(results.boxes) == 0:
            return sv.Detections.empty()
            
        return sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )

    # Decide whether to tile
    use_tiles = False
    if args.mode == "tile":
        use_tiles = True
    elif args.mode == "auto" and long_edge >= args.tile_threshold:
        use_tiles = True

    if use_tiles:
        slicer = sv.InferenceSlicer(
            callback=yolo_predict,
            slice_wh=(args.tile_size, args.tile_size),
            overlap_wh=(args.tile_overlap, args.tile_overlap),
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        )
        detections = slicer(img)
    else:
        detections = yolo_predict(img)

    detected = set()
    if detections.class_id is not None and len(detections.class_id) > 0:
        for cls_id, conf in zip(detections.class_id, detections.confidence):
            if conf >= conf_thresh:
                detected.add(str(model.names[int(cls_id)]))
    return detected


def main():
    args = parse_args()
    
    if args.tile_size <= 0:
        print("Error: tile-size must be positive.", file=sys.stderr)
        return 2
    if args.max_edge <= 0:
        print("Error: max-edge must be positive.", file=sys.stderr)
        return 2
    if not 0 <= args.tile_overlap < args.tile_size:
        print("Error: tile-overlap must be >= 0 and smaller than tile-size.", file=sys.stderr)
        return 2
    if args.tile_threshold <= 0:
        print("Error: tile-threshold must be positive.", file=sys.stderr)
        return 2

    source_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(args.input)))
    output_root = os.path.abspath(os.path.expandvars(os.path.expanduser(args.output)))
    model_path = os.path.abspath(os.path.expandvars(os.path.expanduser(args.model)))

    if not 0.0 <= args.conf <= 1.0:
        print("Error: --conf must be between 0.0 and 1.0.", file=sys.stderr)
        return 2
    if not os.path.isdir(source_dir):
        print(f"Error: source directory does not exist: {source_dir}", file=sys.stderr)
        return 1
    if not os.path.exists(model_path):
        print(f"Error: model path not found: {model_path}", file=sys.stderr)
        return 1
    if is_within(output_root, source_dir):
        print("Error: output directory cannot contain the input directory or be the same directory.", file=sys.stderr)
        return 2

    bin_path = os.path.splitext(model_path)[0] + ".bin"
    if not os.path.isfile(bin_path) and os.path.isdir(model_path):
        bin_path = os.path.join(model_path, os.path.basename(model_path) + ".bin")
    
    print(f"--> Loading OpenVINO model: {model_path}")
    try:
        model = YOLO(model_path, task="detect")
    except Exception as exc:
        print(f"Error: could not load model: {exc}", file=sys.stderr)
        return 1

    print(f"--> Scanning images in {source_dir}...")
    all_image_paths = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not is_within(os.path.join(root, d), output_root)]
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                all_image_paths.append(os.path.join(root, filename))

    all_image_paths.sort()
    print(f"--> Found {len(all_image_paths)} images to analyze and sort.")
    if not all_image_paths: return 0

    if not args.dry_run:
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(os.path.join(output_root, "empty_or_unknown"), exist_ok=True)
        
    LOG_PATH = os.path.join(output_root, "failures.csv")
    SUMMARY_PATH = os.path.join(output_root, "run_summary.txt")
    
    class_counts = Counter()
    total_detections = 0
    last_update = time.time()
    pbar = tqdm(all_image_paths, desc="Processing", unit="img")

    for img_path in pbar:
        filename = os.path.basename(img_path)

        ok, why = validate_image(img_path)
        if not ok:
            print(f"\n[skip] {img_path}: {why}", file=sys.stderr)
            log_failure(LOG_PATH, "validate", img_path, why)
            continue

        try:
            detected_objects = detect_with_mode(model, img_path, args.conf, args)
        except Exception as exc:
            print(f"\n[skip] {filename}: inference failed: {exc}", file=sys.stderr)
            log_failure(LOG_PATH, "inference", img_path, str(exc))
            continue

        if detected_objects:
            total_detections += 1
            primary_class = pick_primary(detected_objects)
            target_folder = os.path.join(output_root, primary_class)
            class_counts[primary_class] += 1
        else:
            target_folder = os.path.join(output_root, "empty_or_unknown")
            class_counts["empty_or_unknown"] += 1

        dest_path = make_destination(img_path, source_dir, target_folder)

        if args.dry_run:
            print(f"{os.path.relpath(img_path, source_dir)} -> {os.path.relpath(dest_path, output_root)}")
        else:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Handle duplicate filenames
            if os.path.exists(dest_path):
                stem, ext = os.path.splitext(dest_path)
                counter = 1
                while os.path.exists(f"{stem}_{counter}{ext}"):
                    counter += 1
                dest_path = f"{stem}_{counter}{ext}"

            try:
                shutil.copy2(img_path, dest_path)
            except Exception as exc:
                print(f"[copy-fail] {filename}: {exc}", file=sys.stderr)
                log_failure(LOG_PATH, "copy", img_path, str(exc))

        # Update rolling progress bar
        if time.time() - last_update > 0.5:
            top_classes = ", ".join(f"{name}:{count}" for name, count in class_counts.most_common(3))
            pbar.set_postfix_str(top_classes)
            last_update = time.time()

    # --- FINAL SUMMARY LOG ---
    print("\n-- Class summary --")
    for name, count in class_counts.most_common():
        print(f"  {name:20s} {count}")

    if not args.dry_run:
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(f"Batch Run Summary: {datetime.now().isoformat()}\n")
            f.write(f"Input Directory: {source_dir}\n")
            f.write(f"Output Directory: {output_root}\n")
            f.write(f"Total Images Scanned: {len(all_image_paths)}\n")
            f.write(f"Images with Detections: {total_detections}\n")
            f.write(f"Images without Detections: {class_counts['empty_or_unknown']}\n\n")
            f.write("Detected Classes:\n")
            for name, count in class_counts.most_common():
                if name != "empty_or_unknown":
                    f.write(f"  - {name}: {count}\n")
        print(f"\nRun summary saved to: {SUMMARY_PATH}")

    print(f"\nAll done! Your files are in: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
