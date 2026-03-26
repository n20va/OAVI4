from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

VARIANT = 11
OPERATOR_NAME = "Оператор Круна 3x3"
GRADIENT_FORMULA_NAME = "G = sqrt(Gx^2 + Gy^2)"
IMAGE_INDEX = 4

THRESHOLD = 110

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"
REPORT_PATH = BASE_DIR / "report.md"


def fetch_image_paths(origin: str, sample_id: str) -> list[str]:
    response = requests.get(f"{origin}/api/samples/{sample_id}", timeout=30)
    response.raise_for_status()
    sample_data = response.json()
    return [f"{origin}/images/{page['filename']}" for page in sample_data["pages"]]


def download_image_rgb(image_url: str) -> np.ndarray:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


def save_rgb(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_gray(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L").save(path)


def cleanup_generated_files(directory: Path) -> None:
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_path.unlink()


def rgb_to_grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float64)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return np.clip(gray, 0, 255).round().astype(np.uint8)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    image_f = image.astype(np.float64)
    padded = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    data_min = float(data.min())
    data_max = float(data.max())
    if data_max <= data_min:
        return np.zeros_like(data, dtype=np.uint8)
    normalized = (data - data_min) * 255.0 / (data_max - data_min)
    return np.clip(normalized, 0, 255).round().astype(np.uint8)


def write_report(source_url: str, width: int, height: int) -> None:
    report = f"""# Лабораторная работа №4
## Выделение контуров на изображении

### Вариант {VARIANT}
- Оператор: {OPERATOR_NAME}
- Формула градиента: `{GRADIENT_FORMULA_NAME}`
- Порог бинаризации градиентной матрицы: `T={THRESHOLD}`

### Исходные данные
- Источник: `{source_url}`
- Размер изображения: `{width}x{height}`

### Результаты

#### Исходное
![img](src/source_color.png)

#### Полутоновое
![img](src/grayscale.bmp)

#### Gx
![img](src/gx_norm.bmp)

#### Gy
![img](src/gy_norm.bmp)

#### G
![img](src/g_norm.bmp)

#### Бинаризация
![img](src/g_binary.bmp)

### Вывод
Контуры успешно выделены оператором Круна.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_generated_files(RESULTS_DIR)
    cleanup_generated_files(SRC_DIR)

    image_paths = fetch_image_paths(ORIGIN, SAMPLE_ID)
    source_url = image_paths[IMAGE_INDEX]

    source_rgb = download_image_rgb(source_url)
    gray = rgb_to_grayscale_weighted(source_rgb)

    kx = np.array([
        [17, 61, 17],
        [0, 0, 0],
        [-17, -61, -17]
    ], dtype=np.float64)

    ky = np.array([
        [17, 0, -17],
        [61, 0, -61],
        [17, 0, -17]
    ], dtype=np.float64)

    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)
    g = np.sqrt(gx ** 2 + gy ** 2)

    gx_n = normalize_to_uint8(np.abs(gx))
    gy_n = normalize_to_uint8(np.abs(gy))
    g_n = normalize_to_uint8(g)
    g_bin = np.where(g_n >= THRESHOLD, 255, 0).astype(np.uint8)

    for out_dir in (RESULTS_DIR, SRC_DIR):
        save_rgb(source_rgb, out_dir / "source_color.png")
        save_gray(gray, out_dir / "grayscale.bmp")
        save_gray(gx_n, out_dir / "gx_norm.bmp")
        save_gray(gy_n, out_dir / "gy_norm.bmp")
        save_gray(g_n, out_dir / "g_norm.bmp")
        save_gray(g_bin, out_dir / "g_binary.bmp")

    h, w = gray.shape
    write_report(source_url, w, h)

    print("Готово!")


if __name__ == "__main__":
    main()
