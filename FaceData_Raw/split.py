import os
import random
import shutil

# =========================
# Settings
# =========================
SOURCE_DIR = "D:\\UOM\\Sem 5\\Pattern\\Assignment03\\RocogniX_Classification_Assignment\\FaceData_Raw\\halfres"       # Folder containing all .pgm images
OUTPUT_DIR = "D:\\UOM\\Sem 5\\Pattern\\Assignment03\\RocogniX_Classification_Assignment\\FaceData_Raw\\split"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =========================
# Get all PGM images
# =========================
images = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(".pgm")
]

print(f"Total images: {len(images)}")

# =========================
# Shuffle with fixed seed
# =========================
random.seed(RANDOM_SEED)
random.shuffle(images)

# =========================
# Calculate split sizes
# =========================
n = len(images)

train_end = int(TRAIN_RATIO * n)
val_end = train_end + int(VAL_RATIO * n)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

print(f"Training:   {len(train_images)}")
print(f"Validation: {len(val_images)}")
print(f"Test:       {len(test_images)}")

# =========================
# Create directories
# =========================
for folder in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

# =========================
# Copy images
# =========================
for image in train_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, image),
        os.path.join(OUTPUT_DIR, "train", image)
    )

for image in val_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, image),
        os.path.join(OUTPUT_DIR, "validation", image)
    )

for image in test_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, image),
        os.path.join(OUTPUT_DIR, "test", image)
    )

print("\nDataset split completed.")