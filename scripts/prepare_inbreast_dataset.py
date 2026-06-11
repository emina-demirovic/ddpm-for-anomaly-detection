from pathlib import Path
import random

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm


# Ovo vodi do: C:\Users\Emina\Desktop\diffusion_model
WORK_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = WORK_ROOT / "dataset_raw" / "INBreast"
DICOM_DIR = RAW_ROOT / "AllDICOMs"
MASK_DIR = RAW_ROOT / "extras" / "MassSegmentationMasks"

OUT_ROOT = WORK_ROOT / "datasets" / "INBreast"
TRAIN_HEALTHY_DIR = OUT_ROOT / "train" / "healthy"
TEST_HEALTHY_DIR = OUT_ROOT / "test" / "healthy"
TEST_MASS_DIR = OUT_ROOT / "test" / "mass"
TEST_MASKS_DIR = OUT_ROOT / "test" / "masks"

IMAGE_SIZE = 256
TEST_HEALTHY_RATIO = 0.2
SEED = 42


def get_image_id(path: Path) -> str:
    return path.stem.split("_")[0]


def ensure_dirs():
    for folder in [
        TRAIN_HEALTHY_DIR,
        TEST_HEALTHY_DIR,
        TEST_MASS_DIR,
        TEST_MASKS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = image - image.min()
    if image.max() > 0:
        image = image / image.max()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def load_and_prepare_dicom(dicom_path: Path) -> Image.Image:
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array

    # Za slučaj MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    image = normalize_to_uint8(image)
    image = Image.fromarray(image).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return image


def load_and_prepare_mask(mask_path: Path) -> Image.Image:
    mask = np.array(Image.open(mask_path))

    # Maska je RGBA -> uzmi prvi kanal
    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = (mask > 0).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    return mask


def save_image(img: Image.Image, out_path: Path):
    img.save(out_path)


def main():
    ensure_dirs()

    dicom_files = sorted(DICOM_DIR.glob("*.dcm"))
    mask_files = sorted(MASK_DIR.glob("*_mask.png"))
    mask_ids = {path.stem.replace("_mask", "") for path in mask_files}

    mass_cases = []
    healthy_cases = []

    for dicom_path in dicom_files:
        image_id = get_image_id(dicom_path)
        if image_id in mask_ids:
            mass_cases.append(dicom_path)
        else:
            healthy_cases.append(dicom_path)

    random.seed(SEED)
    random.shuffle(healthy_cases)

    n_test_healthy = int(len(healthy_cases) * TEST_HEALTHY_RATIO)
    if n_test_healthy == 0 and len(healthy_cases) > 0:
        n_test_healthy = 1

    test_healthy_cases = healthy_cases[:n_test_healthy]
    train_healthy_cases = healthy_cases[n_test_healthy:]

    print("=== Preparing INBreast dataset ===")
    print(f"Total DICOMs: {len(dicom_files)}")
    print(f"Mass cases: {len(mass_cases)}")
    print(f"Healthy candidates: {len(healthy_cases)}")
    print(f"Train healthy: {len(train_healthy_cases)}")
    print(f"Test healthy: {len(test_healthy_cases)}")
    print()

    # Train healthy
    for dicom_path in tqdm(train_healthy_cases, desc="Saving train healthy"):
        image_id = get_image_id(dicom_path)
        image = load_and_prepare_dicom(dicom_path)
        save_image(image, TRAIN_HEALTHY_DIR / f"{image_id}.png")

    # Test healthy
    for dicom_path in tqdm(test_healthy_cases, desc="Saving test healthy"):
        image_id = get_image_id(dicom_path)
        image = load_and_prepare_dicom(dicom_path)
        save_image(image, TEST_HEALTHY_DIR / f"{image_id}.png")

    # Test mass + masks
    for dicom_path in tqdm(mass_cases, desc="Saving test mass + masks"):
        image_id = get_image_id(dicom_path)
        mask_path = MASK_DIR / f"{image_id}_mask.png"

        image = load_and_prepare_dicom(dicom_path)
        mask = load_and_prepare_mask(mask_path)

        save_image(image, TEST_MASS_DIR / f"{image_id}.png")
        save_image(mask, TEST_MASKS_DIR / f"{image_id}_mask.png")

    print()
    print("Done.")
    print(f"Saved dataset to: {OUT_ROOT}")


if __name__ == "__main__":
    main()