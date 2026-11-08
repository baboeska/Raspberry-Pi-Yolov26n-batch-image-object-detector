#!/usr/bin/env python3
"""
Batch-sort images into folders based on YOLO26 object detections.

Requires:
    pip install ultralytics openvino opencv-python tqdm

Pass the OpenVINO .xml file to --model. Keep its matching .bin file beside it.
Example:
    python processor_openvino.py -i ~/Pictures -o ~/Pictures_Sorted \
        -m ./yolo26s_openvino_model/yolo26s.xml --conf 0.40
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


# --- CONFIGURATION ---
MODEL_PATH = "~/yolo26n_openvino_model"  # Change this to your actual XML path, or use --model.
#MODEL_PATH = "yolo11s.onnx"
DEFAULT_SOURCE = "~/Pictures"
#DEFAULT_SOURCE = "~/bus"
DEFAULT_OUTPUT = "~/Picturez"
DEFAULT_CONF = 0.40

# If multiple classes are detected, the first matching class here wins.
PRIORITY = [
    "person", "car", "bus", "truck",
    "motorcycle", "bicycle", "train", "boat",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="dsbatch",
        description="Batch-sort images using Ultralytics YOLO26 with an OpenVINO model.",
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_SOURCE,
        help=f"source directory to scan (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"destination root for sorted folders (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=DEFAULT_CONF,
        help=f"detection confidence threshold (default: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--model", "-m", default=MODEL_PATH,
        help=f"path to the OpenVINO .xml model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--device", default=None,
        help="optional inference device, e.g. CPU or GPU; leave unset for Ultralytics default",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="inference image size (default: 640)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print routing decisions without copying files",
    )
    return parser.parse_args()


def pick_primary(detected):
    """Choose the highest-priority detected class."""
    for class_name in PRIORITY:
        if class_name in detected:
            return class_name
    return sorted(detected)[0]


def is_within(path, parent):
    """True when path is parent itself or is below parent."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def make_destination(img_path, source_dir, output_root, target_folder):
    """Preserve subdirectories and disambiguate same-named files from different folders."""
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


def detect_classes(model, image, conf_thresh, imgsz, device=None):
    """
    Run Ultralytics inference and return the set of class names detected.

    Ultralytics handles OpenVINO input preprocessing, YOLO26 output decoding,
    and post-processing/NMS behavior. No manual letterboxing or output decoding
    is needed here.
    """
    predict_args = {
        "source": image,
        "conf": conf_thresh,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        predict_args["device"] = device

    results = model.predict(**predict_args)
    if not results:
        return set()

    result = results[0]
    if result.boxes is None or result.boxes.cls is None:
        return set()

    class_ids = result.boxes.cls.int().cpu().tolist()
    names = model.names

    detected = set()
    for class_id in class_ids:
        if isinstance(names, dict):
            class_name = names.get(class_id)
        else:
            class_name = names[class_id] if 0 <= class_id < len(names) else None

        if class_name is not None:
            detected.add(str(class_name))
    print(f"DEBUG: {len(result.boxes)} boxes, classes={result.boxes.cls.tolist()}")
    return detected


def main():
    args = parse_args()

    source_dir = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.input))
    )
    output_root = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.output))
    )
    model_path = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.model))
    )

    if not 0.0 <= args.conf <= 1.0:
        print("Error: --conf must be between 0.0 and 1.0.", file=sys.stderr)
        return 2

    if not os.path.isdir(source_dir):
        print(f"Error: source directory does not exist: {source_dir}", file=sys.stderr)
        return 1

    if not os.path.exists(model_path):
        print(f"Error: model path not found: {model_path}", file=sys.stderr)
        return 1

    bin_path = os.path.splitext(model_path)[0] + ".bin"
    if not os.path.isfile(bin_path):
        print(
            f"Warning: matching BIN file not found beside XML: {bin_path}\n"
            "OpenVINO normally requires the XML and BIN files together.",
            file=sys.stderr,
        )

    print(f"--> Loading Ultralytics/OpenVINO model: {model_path}")
    try:
        model = YOLO(model_path, task="detect")
    except Exception as exc:
        print(f"Error: could not load OpenVINO model: {exc}", file=sys.stderr)
        return 1

    print(f"--> Scanning images in {source_dir}...")
    all_image_paths = []

    for root, dirs, files in os.walk(source_dir):
        # Do not recursively process the destination if it is inside the source.
        dirs[:] = [
            directory for directory in dirs
            if not is_within(os.path.join(root, directory), output_root)
        ]

        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                all_image_paths.append(os.path.join(root, filename))

    all_image_paths.sort()
    print(f"--> Found {len(all_image_paths)} images to analyze and sort.")

    if not all_image_paths:
        return 0

    if not args.dry_run:
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(os.path.join(output_root, "empty_or_unknown"), exist_ok=True)

    for img_path in tqdm(all_image_paths, desc="Processing Images", unit="img"):
        filename = os.path.basename(img_path)

        image = cv2.imread(img_path)
        if image is None:
            print(f"\n[skip] Could not read image: {img_path}", file=sys.stderr)
            continue

        try:
            detected_objects = detect_classes(
                model=model,
                image=image,
                conf_thresh=args.conf,
                imgsz=args.imgsz,
                device=args.device,
            )
        except Exception as exc:
            print(f"\n[skip] {filename}: inference failed: {exc}", file=sys.stderr)
            continue

        if detected_objects:
            primary_class = pick_primary(detected_objects)
            target_folder = os.path.join(output_root, primary_class)
        else:
            target_folder = os.path.join(output_root, "empty_or_unknown")

        dest_path = make_destination(
            img_path, source_dir, output_root, target_folder
        )
        relative_destination = os.path.relpath(dest_path, output_root)

        if args.dry_run:
            print(f"{os.path.relpath(img_path, source_dir)} -> {relative_destination}")
            continue

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            shutil.copy2(img_path, dest_path)
        except Exception as exc:
            print(f"[copy-fail] {filename}: {exc}", file=sys.stderr)

    print(f"\nAll done! Your files are in: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
