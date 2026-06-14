"""Загрузка таблиц H&M через Kaggle CLI.

Файлы скачиваются по отдельности, поэтому архив с изображениями не загружается.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

_AUTH_HINT = (
    "\n[prep] Не удалось скачать с Kaggle. Проверь:\n"
    "  1) положи kaggle.json в ~/.kaggle/ (chmod 600 ~/.kaggle/kaggle.json);\n"
    "  2) прими правила соревнования на\n"
    "     https://www.kaggle.com/competitions/"
    "h-and-m-personalized-fashion-recommendations/rules\n"
    "Перед загрузкой нужно принять правила соревнования.\n"
)


def _unzip_if_needed(raw_dir: Path, filename: str) -> None:
    """Распаковать архив, если CSV ещё не появился."""
    target = raw_dir / filename
    zip_path = raw_dir / f"{filename}.zip"
    if target.exists() or not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()


def _download_file(competition: str, filename: str, raw_dir: Path) -> None:
    """Скачать один файл, если его ещё нет локально."""
    target = raw_dir / filename
    if target.exists():
        print(f"[prep] {filename} уже есть — пропускаю загрузку")
        return

    print(f"[prep] downloading {filename} from {competition} ...")
    cmd = [
        "kaggle", "competitions", "download",
        "-c", competition,
        "-f", filename,
        "-p", str(raw_dir),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "kaggle CLI не найден. Установи: pip install -r requirements.txt"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_AUTH_HINT) from exc

    _unzip_if_needed(raw_dir, filename)
    if not target.exists():
        raise RuntimeError(f"[prep] {filename} не появился после загрузки/распаковки")


def ensure_raw_data(competition: str, files: list[str], raw_dir: Path) -> None:
    """Скачать отсутствующие файлы в raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Сначала проверяем доступ на небольшом articles.csv.
    ordered = sorted(files, key=lambda f: 0 if "articles" in f else 1)
    for filename in ordered:
        _download_file(competition, filename, raw_dir)
