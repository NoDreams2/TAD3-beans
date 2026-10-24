"""Стадия train: обучает одну модель (имя передаётся аргументом --model).
DVC запускает этот скрипт для каждой модели из params.yaml -> models.

Во время обучения после каждой эпохи модель проверяется на validation.
Сохраняется состояние модели на лучшей эпохе (с минимальной ошибкой на
проверке), а кривые обучения пишутся в plots/train/<модель>.csv."""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import make_loader
from models import build_model, count_params
from utils import AUGMENTED_DIR, PROCESSED_DIR, get_device, load_params, save_json, set_seed


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    name = parser.parse_args().model

    params = load_params()
    mp, tp = params["models"][name], params["train"]
    set_seed(params["seed"])
    device = get_device()
    use_aug = tp.get("use_augmented", False)
    train_dir = (AUGMENTED_DIR if use_aug else PROCESSED_DIR) / "train"

    gen = torch.Generator().manual_seed(params["seed"])
    train_dl = make_loader(train_dir, tp["batch_size"], tp["num_workers"], True, gen)
    val_dl = make_loader(PROCESSED_DIR / "validation", tp["batch_size"], tp["num_workers"], False)
    classes = train_dl.dataset.classes

    model = build_model(name, len(classes), mp["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mp["lr"], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    print(f"{name}: {count_params(model):,} параметров, устройство {device}, "
          f"обучающих картинок {len(train_dl.dataset)} ({'с аугментацией' if use_aug else 'исходные'})")

    history, best, start = [], None, time.time()
    for epoch in range(1, mp["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "val_loss": round(va_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4)})
        mark = ""
        if best is None or va_loss < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": va_loss, "val_acc": va_acc}
            Path("models").mkdir(exist_ok=True)
            torch.save({"model": name, "state_dict": model.state_dict(), "classes": classes,
                        "best_epoch": epoch}, f"models/{name}.pt")
            mark = "  <- лучшая, сохранена"
        print(f"  эпоха {epoch:2d}/{mp['epochs']}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"val_acc={va_acc:.4f}{mark}")

    plot_path = Path(f"plots/train/{name}.csv")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    with plot_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    save_json({
        "best_epoch": best["epoch"],
        "best_val_loss": round(best["val_loss"], 4),
        "best_val_acc": round(best["val_acc"], 4),
        "epochs": mp["epochs"],
        "train_images": len(train_dl.dataset),
        "params": count_params(model),
        "train_time_sec": round(time.time() - start, 1),
    }, f"metrics/train/{name}.json")
    print(f"{name}: лучшая эпоха {best['epoch']}, точность на проверке {best['val_acc']:.4f}")


if __name__ == "__main__":
    main()
