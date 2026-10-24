"""Стадия prepare (предобработка): приводит все картинки к одному размеру
img_size x img_size и сохраняет в data/processed/. Обучение потом читает
маленькие файлы, это быстрее, чем каждый раз уменьшать исходные 500x500."""
import shutil

from PIL import Image

from utils import PROCESSED_DIR, RAW_DIR, load_params


def main():
    size = load_params()["data"]["img_size"]
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    count = 0
    for src in sorted(RAW_DIR.rglob("*.jpg")):
        dst = PROCESSED_DIR / src.relative_to(RAW_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").resize((size, size), Image.BICUBIC).save(dst, quality=95)
        count += 1
    print(f"Подготовлено {count} картинок размером {size}x{size}")


if __name__ == "__main__":
    main()
