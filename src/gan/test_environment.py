import sys
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image


print("=" * 50)
print("SYNTHETIC MEDICAL DATA GENERATOR")
print("ENVIRONMENT TEST")
print("=" * 50)

print("Python version:", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("Pillow: OK")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not available - using CPU")

print("=" * 50)
print("ENVIRONMENT READY")
print("=" * 50)