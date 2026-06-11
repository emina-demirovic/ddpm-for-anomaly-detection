from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def list_png_files(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Required folder does not exist: {folder}")

    return sorted(folder.glob("*.png"))


def check_folder(folder: Path, data_root: Path, expected_size: int) -> None:
    files = list_png_files(folder)
    sizes = set()

    for path in files:
        with Image.open(path) as image:
            sizes.add(image.size)

    relative_folder = folder.relative_to(data_root)
    print(f"{relative_folder}: {len(files)} files, sizes: {sizes}")

    expected = (expected_size, expected_size)
    if sizes and sizes != {expected}:
        raise ValueError(
            f"Unexpected image size in {relative_folder}. "
            f"Expected {expected}, found {sizes}."
        )


def check_mass_mask_pairs(test_mass_dir: Path, test_masks_dir: Path) -> None:
    mass_ids = {path.stem for path in test_mass_dir.glob("*.png")}
    mask_ids = {
        path.stem.replace("_mask", "")
        for path in test_masks_dir.glob("*_mask.png")
    }

    missing_masks = mass_ids - mask_ids
    masks_without_image = mask_ids - mass_ids

    print()
    print(f"Mass images without mask: {len(missing_masks)}")
    print(f"Masks without mass image: {len(masks_without_image)}")

    if missing_masks:
        print("First missing masks:")
        print(sorted(missing_masks)[:10])

    if masks_without_image:
        print("First masks without image:")
        print(sorted(masks_without_image)[:10])

    if missing_masks or masks_without_image:
        raise ValueError("Mass images and segmentation masks do not match.")


def check_blank_masks(test_masks_dir: Path) -> None:
    blank_masks = []

    for mask_path in sorted(test_masks_dir.glob("*_mask.png")):
        mask = np.array(Image.open(mask_path))

        if mask.max() == 0:
            blank_masks.append(mask_path.name)

    print(f"Blank masks after resize: {len(blank_masks)}")

    if blank_masks:
        print("First blank masks:")
        print(blank_masks[:10])
        raise ValueError("Blank masks found in the prepared dataset.")


def check_dataset(data_root: Path, image_size: int) -> None:
    train_healthy_dir = data_root / "train" / "healthy"
    test_healthy_dir = data_root / "test" / "healthy"
    test_mass_dir = data_root / "test" / "mass"
    test_masks_dir = data_root / "test" / "masks"

    print("=== Checking prepared INBreast dataset ===")
    print(f"Dataset root: {data_root}")
    print()

    check_folder(train_healthy_dir, data_root, image_size)
    check_folder(test_healthy_dir, data_root, image_size)
    check_folder(test_mass_dir, data_root, image_size)
    check_folder(test_masks_dir, data_root, image_size)

    check_mass_mask_pairs(test_mass_dir, test_masks_dir)
    check_blank_masks(test_masks_dir)

    print()
    print("Dataset validation passed.")


def parse_args():
    parser = ArgumentParser(description="Validate the prepared INBreast dataset.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets") / "INBreast",
        help="Path to the prepared INBreast dataset.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Expected image and mask size.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    check_dataset(data_root=args.data_root, image_size=args.image_size)


if __name__ == "__main__":
    main()