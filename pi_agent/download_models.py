from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

from .config import load_config


MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}


def download_model(filename: str, target: Path, force: bool) -> None:
    if target.exists() and not force:
        print(f"Already exists: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.download")

    for url in MODEL_URLS[filename]:
        try:
            print(f"Downloading {filename}")
            print(f"  {url}")
            urlretrieve(url, partial)
            if partial.stat().st_size < 1024:
                raise RuntimeError(f"Downloaded file is too small: {partial}")
            partial.replace(target)
            print(f"Saved {target}")
            return
        except (OSError, RuntimeError, URLError) as exc:
            if partial.exists():
                partial.unlink()
            print(f"Failed: {exc}")

    raise RuntimeError(f"Could not download {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenCV face models.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="Path to pi_agent/config.json",
    )
    parser.add_argument("--force", action="store_true", help="Download even if files already exist")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    download_model("face_detection_yunet_2023mar.onnx", config.detector_model_path, args.force)
    download_model("face_recognition_sface_2021dec.onnx", config.recognition_model_path, args.force)


if __name__ == "__main__":
    main()
