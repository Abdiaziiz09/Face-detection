import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

MODELS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "desc": "YuNet Face Detector"
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "desc": "SFace Face Recognizer"
    }
}

def download_file(url: str, dest_path: Path, desc: str):
    """Downloads a model file with progress tracking if not already present."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[OK] {desc} already downloaded: {dest_path.name}")
        return

    print(f"[+] Downloading {desc} ({dest_path.name})...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    
    response = requests.get(url, stream=True, allow_redirects=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(temp_path, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=dest_path.name, ncols=80
    ) as pbar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    temp_path.replace(dest_path)
    print(f"[Done] {dest_path.name} downloaded successfully!\n")

def ensure_models(models_dir: str = "models"):
    """Ensures all required open-source ONNX models exist locally."""
    target_dir = Path(models_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for key, info in MODELS.items():
        file_path = target_dir / info["filename"]
        download_file(info["url"], file_path, info["desc"])
        paths[key] = str(file_path)
    return paths

if __name__ == "__main__":
    ensure_models()
