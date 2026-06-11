from argparse import ArgumentParser
from pathlib import Path
import random
import shutil

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm


def get_image_id(path: Path) -> str:
    """
    Extracts the INBreast image identifier from a DICOM filename.
    Example:
        20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm -> 20586908
    """
    return path.stem.split("_")[0]


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalizes a DICOM pixel array to the uint8 range [0, 255].
    """
    image = image.astype(np.float32)
    image = image - image.min()

    if image.max() > 0:
        image = image / image.max()

    return (image * 255).clip(0, 255).astype(np.uint8)


def load_and_prepare_dicom(dicom_path: Path, image_size: int) -> Image.Image:
    """
    Loads a DICOM image, handles MONOCHROME1 interpretation if needed,
    converts it to grayscale and resizes it.
    """
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array

    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    image = normalize_to_uint8(image)
    image = Image.fromarray(image).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    return image


def load_and_prepare_mask(mask_path: Path, image_size: int) -> Image.Image:
    """
    Loads a binary mass segmentation mask and resizes it using nearest-neighbor
    interpolation to preserve binary labels.
    """
    mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = (mask > 0).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)

    return mask


def prepare_dataset(
    raw_root: Path,
    output_root: Path,
    image_size: int,
    test_healthy_ratio: float,
    seed: int,
    overwrite: bool,
) -> None:
    dicom_dir = raw_root / "AllDICOMs"
    mask_dir = raw_root / "extras" / "MassSegmentationMasks"

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    train_healthy_dir = output_root / "train" / "healthy"
    test_healthy_dir = output_root / "test" / "healthy"
    test_mass_dir = output_root / "test" / "mass"
    test_masks_dir = output_root / "test" / "masks"

    ensure_clean_dir(output_root, overwrite)
    train_healthy_dir.mkdir(parents=True, exist_ok=True)
    test_healthy_dir.mkdir(parents=True, exist_ok=True)
    test_mass_dir.mkdir(parents=True, exist_ok=True)
    test_masks_dir.mkdir(parents=True, exist_ok=True)

    dicom_files = sorted(dicom_dir.glob("*.dcm"))
    mask_files = sorted(mask_dir.glob("*_mask.png"))
    mask_ids = {path.stem.replace("_mask", "") for path in mask_files}

    mass_cases = []
    healthy_cases = []

    for dicom_path in dicom_files:
        image_id = get_image_id(dicom_path)

        if image_id in mask_ids:
            mass_cases.append(dicom_path)
        else:
            healthy_cases.append(dicom_path)

    random.seed(seed)
    random.shuffle(healthy_cases)

    n_test_healthy = int(len(healthy_cases) * test_healthy_ratio)

    if n_test_healthy == 0 and len(healthy_cases) > 0:
        n_test_healthy = 1

    test_healthy_cases = healthy_cases[:n_test_healthy]
    train_healthy_cases = healthy_cases[n_test_healthy:]

    print("=== Preparing INBreast dataset ===")
    print(f"Raw root: {raw_root}")
    print(f"Output root: {output_root}")
    print(f"Total DICOM files: {len(dicom_files)}")
    print(f"Mass cases: {len(mass_cases)}")
    print(f"Healthy cases: {len(healthy_cases)}")
    print(f"Train healthy: {len(train_healthy_cases)}")
    print(f"Test healthy: {len(test_healthy_cases)}")
    print()

    for dicom_path in tqdm(train_healthy_cases, desc="Saving train healthy"):
        image_id = get_image_id(dicom_path)
        image = load_and_prepare_dicom(dicom_path, image_size)
        image.save(train_healthy_dir / f"{image_id}.png")

    for dicom_path in tqdm(test_healthy_cases, desc="Saving test healthy"):
        image_id = get_image_id(dicom_path)
        image = load_and_prepare_dicom(dicom_path, image_size)
        image.save(test_healthy_dir / f"{image_id}.png")

    for dicom_path in tqdm(mass_cases, desc="Saving test mass and masks"):
        image_id = get_image_id(dicom_path)
        mask_path = mask_dir / f"{image_id}_mask.png"

        image = load_and_prepare_dicom(dicom_path, image_size)
        mask = load_and_prepare_mask(mask_path, image_size)

        image.save(test_mass_dir / f"{image_id}.png")
        mask.save(test_masks_dir / f"{image_id}_mask.png")

    print()
    print("Done.")
    print(f"Prepared dataset saved to: {output_root}")


def parse_args():
    parser = ArgumentParser(description="Prepare the INBreast dataset for DDPM anomaly detection.")

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("dataset_raw") / "INBreast",
        help="Path to the raw INBreast dataset folder.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets") / "INBreast",
        help="Path where the prepared INBreast dataset will be saved.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Output image and mask size.",
    )

    parser.add_argument(
        "--test-healthy-ratio",
        type=float,
        default=0.2,
        help="Ratio of healthy cases used for the test split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the healthy train/test split.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing prepared dataset before creating a new one.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    prepare_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        image_size=args.image_size,
        test_healthy_ratio=args.test_healthy_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()