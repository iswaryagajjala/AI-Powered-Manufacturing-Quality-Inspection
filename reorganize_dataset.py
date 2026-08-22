import shutil
from pathlib import Path

# Your exact extracted location
SOURCE_DIR = Path(r"C:\Users\Likhitha\Downloads\archive (1)\NEU-DET")

DEST_DIR = Path("data/raw/NEU-CLS")
DEST_DIR.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

for split_name in ("train", "validation"):
    images_dir = SOURCE_DIR / split_name / "images"
    if not images_dir.exists():
        print(f"Skipping missing: {images_dir}")
        continue

    for class_folder in images_dir.iterdir():
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        dest_class_dir = DEST_DIR / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        for img in class_folder.glob("*"):
            if img.suffix.lower() in image_exts:
                dest_path = dest_class_dir / f"{split_name}_{img.name}"
                shutil.copy2(img, dest_path)

print("\nDone. Image counts per class:")
for cls_dir in sorted(DEST_DIR.iterdir()):
    if cls_dir.is_dir():
        count = len(list(cls_dir.glob("*")))
        print(f"  {cls_dir.name}: {count}")