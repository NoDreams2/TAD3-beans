"""Стадия download: скачивает датасет Beans с Hugging Face и раскладывает
картинки по папкам data/raw/<часть>/<класс>/.

Части уже заданы авторами датасета: train (обучение), validation (проверка
во время обучения), test (итоговая оценка)."""
import io
import json
import shutil

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from utils import RAW_DIR, load_params

SPLITS = ["train", "validation", "test"]


def class_names(table) -> list:
    """Названия классов берём из метаданных parquet-файла (их записал Hugging Face)."""
    meta = json.loads(table.schema.metadata[b"huggingface"])
    names = meta["info"]["features"]["labels"]["names"]
    return list(names.values()) if isinstance(names, dict) else names


def main():
    p = load_params()["data"]
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    for split in SPLITS:
        path = hf_hub_download(p["hf_dataset"], f"data/{split}-00000-of-00001.parquet",
                               repo_type="dataset", revision=p["revision"])
        table = pq.read_table(path)
        names = class_names(table)
        rows = table.to_pylist()
        for row in rows:
            label = names[row["labels"]]
            out_dir = RAW_DIR / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            name = row["image_file_path"].replace("\\", "/").split("/")[-1]
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            img.save(out_dir / name, quality=95)
        print(f"{split}: {len(rows)} картинок, классы: {names}")


if __name__ == "__main__":
    main()
