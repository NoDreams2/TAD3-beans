"""Стадия evaluate: оценивает все обученные модели на тестовой части,
которую модели не видели ни при обучении, ни при выборе лучшей эпохи.

Пишет metrics/test.json (метрики), plots/confusion/<модель>.csv (для матрицы
ошибок) и reports/summary.md (сводная таблица)."""
import csv
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data import make_loader
from models import build_model
from utils import PROCESSED_DIR, get_device, load_params, save_json


def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_pred += model(x.to(device)).argmax(1).cpu().tolist()
            y_true += y.tolist()
    return y_true, y_pred


def main():
    params = load_params()
    tp = params["train"]
    device = get_device()
    loader = make_loader(PROCESSED_DIR / "test", tp["batch_size"], tp["num_workers"], False)
    classes = loader.dataset.classes
    results, rows = {}, []
    for name in params["models"]:
        ckpt = torch.load(f"models/{name}.pt", map_location=device)
        model = build_model(name, len(classes), pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
        y_true, y_pred = predict(model, loader, device)
        m = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "best_epoch": ckpt["best_epoch"],
        }
        results[name] = m
        path = Path(f"plots/confusion/{name}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["actual", "predicted"])
            w.writerows([classes[a], classes[b]] for a, b in zip(y_true, y_pred))
        rows.append(f"| {name} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {m['best_epoch']} |")
        print(f"{name}: accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")

    save_json(results, "metrics/test.json")
    data_kind = "исходные + аугментированные" if tp.get("use_augmented") else "исходные"
    Path("reports").mkdir(exist_ok=True)
    Path("reports/summary.md").write_text(
        f"# Результаты на тестовой части ({len(loader.dataset)} фото)\n\n"
        f"Обучающие данные: {data_kind}\n\n"
        "| Модель | Accuracy | F1 (macro) | Лучшая эпоха |\n|---|---:|---:|---:|\n"
        + "\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
