from pathlib import Path
import math

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "synthetic" / "generated"


# ============================================================
# FIND SYNTHETIC IMAGES
# ============================================================

image_paths = sorted(
    IMAGE_DIR.glob("synthetic_xray_[0-9][0-9][0-9].png")
)

if len(image_paths) == 0:
    print("No synthetic images found.")
    exit()


# Use first 25 images
image_paths = image_paths[:25]

print("=" * 60)
print("SYNTHETIC X-RAY VISUAL DIVERSITY")
print("=" * 60)

print(f"\nImages displayed: {len(image_paths)}")


# ============================================================
# CREATE GRID
# ============================================================

cols = 5
rows = math.ceil(len(image_paths) / cols)

plt.figure(figsize=(12, 12))

for i, image_path in enumerate(image_paths):

    image = Image.open(image_path).convert("L")

    plt.subplot(rows, cols, i + 1)

    plt.imshow(image, cmap="gray")

    plt.title(image_path.stem, fontsize=8)

    plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE RESULT
# ============================================================

OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "samples" / "diversity_visual.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nVisual diversity grid saved to:")
print(OUTPUT_PATH)

print("\nDIVERSITY VISUAL TEST COMPLETE")