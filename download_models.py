"""
Download YuNet and SFace ONNX models from OpenCV Zoo.
Run this once to set up the models folder.
"""
import requests
import sys
from pathlib import Path

MODELS = [
    {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "desc": "YuNet Face Detector",
        "min_size": 100_000,
    },
    {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "desc": "SFace Face Recognizer",
        "min_size": 100_000,
    },
]

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download(url: str, dest: Path, desc: str) -> bool:
    print(f"\n[Downloading] {desc}")
    print(f"  URL : {url}")
    print(f"  Dest: {dest}")
    try:
        resp = requests.get(url, stream=True, allow_redirects=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        print(f"  Size: {total:,} bytes" if total else "  Size: unknown")

        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = int(pct / 5) * "█"
                        print(f"\r  [{bar:<20}] {pct:.1f}%", end="", flush=True)
        print(f"\n  [OK] {downloaded:,} bytes written.")
        return True
    except Exception as exc:
        print(f"\n  [ERROR] {exc}")
        dest.unlink(missing_ok=True)
        return False


def main():
    print("=" * 55)
    print("  FaceEngine AI — Model Setup")
    print("=" * 55)

    all_ok = True
    for m in MODELS:
        dest = MODELS_DIR / m["filename"]
        if dest.exists() and dest.stat().st_size >= m["min_size"]:
            print(f"\n[SKIP] {m['desc']} already present ({dest.stat().st_size:,} bytes)")
            continue
        ok = download(m["url"], dest, m["desc"])
        if not ok:
            all_ok = False

    print("\n" + "=" * 55)
    print("  Verifying models with OpenCV ...")
    print("=" * 55)
    try:
        import cv2
        yunet_path = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
        sface_path = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")
        cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
        print("  YuNet detector  : OK")
        cv2.FaceRecognizerSF.create(sface_path, "")
        print("  SFace recognizer: OK")
        print("\n  All models are ready — you can now run the app!")
    except Exception as exc:
        print(f"\n  [VERIFICATION ERROR] {exc}")
        all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
