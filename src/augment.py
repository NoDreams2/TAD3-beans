"""Стадия augment: из каждой обучающей картинки делает `copies` новых
с помощью случайных преобразований и сохраняет новый датасет в data/augmented/.

В новый датасет входят исходные обучающие картинки и их изменённые копии.
Проверочная и тестовая части НЕ изменяются: модель всегда проверяется на
настоящих фото, иначе оценка была бы завышена."""
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from utils import AUGMENTED_DIR, PROCESSED_DIR, load_params, set_seed


def build_transform(p: dict, size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=tuple(p["crop_scale"]), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p["flip"]),
        transforms.RandomRotation(p["rotation"], fill=0),
        transforms.ColorJitter(brightness=p["brightness"], contrast=p["contrast"]),
    ])


def save_examples(pairs, path: Path, cell: int = 160):
    """Картинка для отчёта: в каждой строке исходное фото и его копии."""
    cols = len(pairs[0])
    sheet = Image.new("RGB", (cols * cell, len(pairs) * cell), "white")
    for r, row in enumerate(pairs):
        for c, img in enumerate(row):
            sheet.paste(img.resize((cell - 4, cell - 4)), (c * cell + 2, r * cell + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def main():
    params = load_params()
    p, size = params["augment"], params["data"]["img_size"]
    set_seed(params["seed"])
    transform = build_transform(p, size)
    if AUGMENTED_DIR.exists():
        shutil.rmtree(AUGMENTED_DIR)
    src_root, dst_root = PROCESSED_DIR / "train", AUGMENTED_DIR / "train"
    examples, originals, created = [], 0, 0
    for src in sorted(src_root.rglob("*.jpg")):
        dst_dir = dst_root / src.parent.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        img.save(dst_dir / src.name, quality=95)
        originals += 1
        row = [img]
        for i in range(p["copies"]):
            new = transform(img)
            new.save(dst_dir / f"{src.stem}_aug{i}.jpg", quality=95)
            row.append(new)
            created += 1
        if originals % 260 == 1 and len(examples) < 4:
            examples.append(row)
    if examples and p["copies"] > 0:
        save_examples(examples, Path("reports/augment_examples.jpg"))
    print(f"Исходных: {originals}, создано новых: {created}, всего в обучающей части: {originals + created}")


if __name__ == "__main__":
    main()
