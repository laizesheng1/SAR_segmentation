import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None
def split_large_image(image: Image.Image, patch_size: int = 256, stride: int = 256):
    ''' Split one large image into patches '''
    w, h = image.size
    patches = []
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            box = (left, top, left + patch_size, top + patch_size)
            patch = image.crop(box)
            if patch.size == (patch_size, patch_size):  # drop incomplete patch
                patches.append(patch)
    return patches

def process_dataset(image_dir, label_dir, output_image_dir, output_label_dir,
                    patch_size=256, train_ratio=0.8):

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))])

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    all_image_patches = []
    all_label_patches = []

    print("Splitting large images into patches...")
    for filename in tqdm(image_files):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)  # assumes same filename

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        image_patches = split_large_image(image, patch_size)
        label_patches = split_large_image(label, patch_size)

        all_image_patches.extend(image_patches)
        all_label_patches.extend(label_patches)

    assert len(all_image_patches) == len(all_label_patches)

    print("Splitting into train and test sets...")
    indices = list(range(len(all_image_patches)))
    train_idx, test_idx = train_test_split(indices, train_size=train_ratio, random_state=42, shuffle=True)

    def save_patches(indices, prefix):
        sub_image_dir = os.path.join(output_image_dir, prefix)
        sub_label_dir = os.path.join(output_label_dir, prefix)
        os.makedirs(sub_image_dir, exist_ok=True)
        os.makedirs(sub_label_dir, exist_ok=True)

        for i in indices:
            img = all_image_patches[i]
            lbl = all_label_patches[i]
            img.save(os.path.join(sub_image_dir, f"{prefix}_{i:06d}.png"))
            lbl.save(os.path.join(sub_label_dir, f"{prefix}_{i:06d}.png"))

    save_patches(train_idx, "train")
    save_patches(test_idx, "test")
    print(f"Saved {len(train_idx)} training and {len(test_idx)} test patches.")



process_dataset(
    image_dir="../sar_lab_source/sar_source",
    label_dir="../sar_lab_source/lab_source",
    output_image_dir="../output/images",
    output_label_dir="../output/labels",
    patch_size=256,
    train_ratio=0.8
)

