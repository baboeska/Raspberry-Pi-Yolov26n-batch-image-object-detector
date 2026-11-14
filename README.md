# BIPOD-Pi

Batch image analysis and object identification processor for Raspberry Pi 4/5.

Intended to make life easier for MotionEye users and those with large image
archives they'd like to rapidly sort into object folders.

Part of [AI BIPOD](https://github.com/baboeska/bipod)

## Performance

| Platform | Model | Throughput |
|---|---|---|
| Raspberry Pi 5 | YOLO26n + OpenVINO | ~10 FPS (estimated) |
| Raspberry Pi 4 | YOLO26n + OpenVINO | ~3–5 FPS (estimated) |

## Status
 Pi 5 (YOLO26n). Should work on Pi 4 with lower throughput.
Variants run on Rockchip and Intel/AMD and NVIDIA tensorRT — if you hit issues, please raise
an issue/discussion saying what you needed to do to make it work.

## Requirements

- Raspberry Pi 4 or 5 (64-bit OS — Pi OS Bookworm or Ubuntu 24.04)
- Python 3.11+ (Bookworm ships 3.11; Ubuntu 24.04 ships 3.12)
- `ultralytics`, `openvino`, `opencv-python`, `tqdm`
- SD card or SSD with a few GB free for the model

## Setup

`bash
# Install Python 3 and venv support
    sudo apt install -y python3-full python3-venv

# Create and activate virtual environment
    python3 -m venv ~/openvino-env
    source ~/openvino-env/bin/activate

# Install core dependencies
    pip install --upgrade pip
    pip install ultralytics openvino opencv-python tqdm
    
    python3 -m pip install --upgrade \
    ultralytics \
    "ultralytics[export-openvino]" \
    openvino \
    opencv-python \
    pillow \
    tqdm \
    supervision




Export the model

The .pt → OpenVINO export can be done on the Pi itself (slower) or on a
more powerful machine and copied over (faster). The output is architecture-
agnostic, so export on a desktop and scp the directory across if you prefer.

    python3 - <<'EOF'
    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")
    model.export(format="openvino", imgsz=640, nms=False)
    EOF

You'll get a directory called yolo26n_openvino_model/ containing
yolo26n.xml, yolo26n.bin, and metadata.yaml.

nms=False matters — without it you'll hit the mask-shape error.
And point --model at the directory, not the .xml.
Usage

# Dry run (validate on a small sample first)
    python bipod-pi.py \
    -i ~/Pictures -o ~/Pictures_Sorted \
    -m ./yolo26n_openvino_model/ --dry-run

# Real run
    pythonbipod-pi.py \
    -i ~/Pictures -o ~/Pictures_Sorted \
    -m ./yolo26n_openvino_model/ --conf 0.4

At the end of a run you get a class summary and a failures.csv log in the
output directory.
Known pitfalls

    Pi 5 thermal throttling. A long batch will hit the thermal limit without
    a heatsink or fan. Throughput drops from ~10 FPS to ~5 FPS once throttled.
    A heatsink + fan is worth it.

    Use a fast SD card or USB SSD. A slow card caps throughput at ~4–5 FPS
    regardless of the model.

    --model points at the directory, not the .xml file inside it.

    nms=False on export — otherwise you get the mask-shape error.

    INT8 quantization on Pi is simulated, not accelerated. OpenVINO on Arm
    runs quantized models in FP32 simulation — you won't see the speedup you
    might expect from a desktop. FP32 is fine for this workload.

    Don't use a 32-bit OS. Ultralytics and OpenVINO need 64-bit.

Licensing

Split-license model:

    Code (Python scripts) → AGPL-3.0 (see LICENSE)

    YOLO model weights → AGPL-3.0 (see Ultralytics' licensing page)

Model files are not included; users export or download them separately and
are responsible for AGPL-3.0 compliance when they use or redistribute them.

No warranty at all. If you use or modify this code, please share it further
via GitHub or other means. also, do good things.

If you have issues, consult your favourite AI code helper — ideally this should
be working for you in half an hour or less.

Thanks and regards to DeepSeek AI, ChatGPT, and Google Search AI, all of which
contributed to and helped make this possible.
