from pathlib import Path

import numpy as np
from PIL import Image


WORK_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = WORK_ROOT / "datasets" / "INBreast"

TRAIN_HEALTHY = DATA_ROOT / "train" / "healthy"
TEST_HEALTHY = DATA_ROOT / "test" / "healthy"
TEST_MASS = DATA_ROOT / "test" / "mass"
TEST_MASKS = DATA_ROOT / "test" / "masks"


def check_folder(folder: Path):
    files = sorted(folder.glob("*.png"))
    sizes = set()

    for path in files:
        img = Image.open(path)
        sizes.add(img.size)

    print(f"{folder.relative_to(DATA_ROOT)}: {len(files)} files, sizes: {sizes}")


def main():
    print("=== Checking prepared INBreast dataset ===")

    check_folder(TRAIN_HEALTHY)
    check_folder(TEST_HEALTHY)
    check_folder(TEST_MASS)
    check_folder(TEST_MASKS)

    mass_ids = {p.stem for p in TEST_MASS.glob("*.png")}
    mask_ids = {p.stem.replace("_mask", "") for p in TEST_MASKS.glob("*.png")}

    print()
    print(f"Mass images without mask: {len(mass_ids - mask_ids)}")
    print(f"Masks without mass image: {len(mask_ids - mass_ids)}")

    blank_masks = []
    for mask_path in sorted(TEST_MASKS.glob("*.png")):
        mask = np.array(Image.open(mask_path))
        if mask.max() == 0:
            blank_masks.append(mask_path.name)

    print(f"Blank masks after resize: {len(blank_masks)}")

    if blank_masks:
        print("First blank masks:")
        print(blank_masks[:10])


if __name__ == "__main__":
    main()