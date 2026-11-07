#Install python 3
sudo apt install python3-full
# Create and activate virtual environment

python3 -m venv openvino-env
source openvino-env/bin/activate

# Install core dependencies (note: INT8 quantization requires nncf)
pip install ultralytics openvino opencv-python tqdm
# If you need INT8 acceleration, additionally install:
# pip install nncf

from ultralytics import YOLO

# Download and export YOLO26s (disable end-to-end to ensure output format compatibility)
model = YOLO("yolo26n.pt")
model.export(format="openvino", imgsz=640, nms=False)
# After export, you will get a directory like yolo26n_openvino_model/


# Dry run test (recommended to validate on a small sample first)
python processor_openvinoraspberry.py -i ~/Pictures -o ~/Pictures_Sorted -m ./yolo26s_openvino_model/ --dry-run

# Formal run
python processor_openvinoraspberry.py -i ~/Pictures -o ~/Pictures_Sorted -m ./yolo26s_openvino_model/ --conf 0.4

#Any glitches? Ask ai to help, chatgpt and deepseek are both awesome.