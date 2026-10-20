# ⚡ FaceEngine AI — Offline Face Recognition & Attendance System

[![Python Version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#system-requirements)
[![Cost](https://img.shields.io/badge/Cost-100%25%20Free%20%26%20Open--Source-brightgreen)](#overview)
[![API Keys](https://img.shields.io/badge/API%20Keys-None%20Required%20(100%25%20Offline)-orange)](#overview)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x%2B-blue?logo=opencv)](https://opencv.org/)
[![Models](https://img.shields.io/badge/Models-YuNet%20%2B%20SFace%20(ONNX)-purple)](#how-it-works-architecture)

A complete, production-ready **Face Recognition and Automated Attendance System** built using state-of-the-art open-source Deep Neural Networks (**OpenCV YuNet + SFace**).

The entire pipeline runs **100% locally and offline** on your computer. It requires **no third-party cloud APIs (no AWS, no Azure, no Face++), no API keys, no internet connection after setup, and zero subscription fees**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [What's New / Changelog](#-whats-new--changelog)
- [System Requirements & Python Compatibility](#-system-requirements--python-compatibility)
- [How It Works (Architecture)](#-how-it-works-architecture)
- [Key Features](#-key-features)
- [Installation & Setup](#-installation--setup)
- [How to Use](#-how-to-use)
  - [1. Modern Desktop GUI (Recommended)](#1-modern-desktop-gui-recommended)
  - [2. Command Line Interface (CLI)](#2-command-line-interface-cli)
- [Project Directory Structure](#-project-directory-structure)
- [Settings & Fine-Tuning](#-settings--fine-tuning)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [License](#-license)

---

## 🔍 Overview

Many modern face recognition solutions rely on proprietary cloud APIs that charge per request, require active internet connectivity, and raise privacy concerns by uploading personal images to remote servers.

**FaceEngine AI** solves this by packaging cutting-edge lightweight deep learning models directly into a local Python application:
* **Private & Secure**: Images and biometric feature vectors never leave your computer.
* **Ultra-Fast Performance**: Achieves 30–60 FPS on standard desktop and laptop CPUs without requiring an expensive GPU.
* **Zero Cost**: Built entirely on free and open-source software (FOSS).
* **No API Keys**: Completely self-contained — nothing to register, no quotas, no rate limits.

---

## 🆕 What's New / Changelog

### v1.1.0 — October 2026
> Model Setup & Interface Overhaul

#### ✅ Added
- **`download_models.py`** — A dedicated, standalone model setup script that downloads both ONNX models (YuNet + SFace) from OpenCV Zoo with a live ASCII progress bar and automatic OpenCV verification. Run once after cloning on any machine.
- **Verified model integrity check** — After download, the script loads both models into OpenCV DNN and confirms they work before exiting. If a download is corrupted or interrupted, it is automatically cleaned up and retried.
- **English-only UI** — All labels, buttons, status messages, and tooltips across the entire Desktop GUI (`gui/app.py`) have been standardized to English.
- **Cleaner GUI layout** — Sidebar navigation, card headers, status bar, and enrollment panel have been polished for a more professional and consistent appearance.
- **`.gitignore` hardened** — Model binaries (`.onnx`), personal face data (`data/known_faces/`), embedding cache (`data/embeddings.pkl`), and attendance logs (`logs/attendance.csv`) are now permanently excluded from version control to protect privacy.
- **`LICENSE`** — MIT License file added to the repository root.
- **`__init__.py`** — Added to `core/` and `gui/` packages for proper Python module imports.

#### 🔧 Fixed
- **Models not loading on fresh clone** — The `models/` folder was empty after cloning because `.onnx` files were excluded by `.gitignore`. This is now resolved: run `python download_models.py` once after cloning to pull both models (~39 MB total) from the official OpenCV Zoo repository.
- **`core/model_manager.py` path resolution** — Updated to use `Path(__file__).parent` for correct relative path resolution regardless of where the script is called from.

#### 🗑️ Removed
- Removed all Somali-language strings from source code, comments, and GUI labels — codebase is now fully English.
- Removed stale test artifacts (`test_yunet.onnx`, `test_system.py` output data) from the repository.

---

## 💻 System Requirements & Python Compatibility

### Supported Python Versions
| Python Version | Status | Notes |
| :--- | :---: | :--- |
| **Python 3.12.x** | ✅ **Recommended** | Fully tested and optimized |
| **Python 3.11.x** | ✅ Supported | Stable and fast |
| **Python 3.10.x** | ✅ Supported | Fully compatible |
| **Python 3.9.x** | ✅ Supported | Fully compatible |
| **Python 3.8.x** | ✅ Supported | Minimum supported version |
| **Python 3.13+** | ✅ Supported | Standard pure-wheel compatibility |

### Supported Operating Systems
* **Windows**: Windows 10 / Windows 11 (64-bit)
* **Linux**: Ubuntu 20.04+, Debian 11+, Fedora, Arch Linux
* **macOS**: macOS 11 (Big Sur) or newer (Intel & Apple Silicon M1/M2/M3)

### Hardware Requirements
* **Processor (CPU)**: Any dual-core or quad-core Intel / AMD / Apple Silicon CPU.
* **Memory (RAM)**: Minimum 2 GB RAM (4 GB+ recommended).
* **Disk Space**: ~150 MB (including models and runtime libraries).
* **Camera**: Any built-in laptop webcam or external USB webcam.

---

## 🧠 How It Works (Architecture)

```
       [ Live Webcam / Image Input ]
                     │
                     ▼
       ┌───────────────────────────┐
       │     YuNet CNN Detector    │  ──▶ Detects bounding box & 5 facial landmarks
       └───────────────────────────┘      (Right eye, Left eye, Nose, Mouth corners)
                     │
                     ▼
       ┌───────────────────────────┐
       │   Affine Landmark Align   │  ──▶ Normalizes rotation & crops face to 112x112
       └───────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │   SFace Feature Network   │  ──▶ Generates 128-dimensional L2-normalized vector
       └───────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Cosine Similarity Matcher │  ──▶ Compares embedding against local database cache
       └───────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   [ Match >= 0.38 ]       [ Match < 0.38 ]
   "Identified Name"          "Unknown"
         │
         ▼
[ Anti-Spam Attendance CSV Log ]
```

1. **Face Detection (YuNet)**:
   YuNet is an ultra-fast, lightweight convolutional network designed by OpenCV Zoo for edge devices. It locates multiple faces and predicts 5 key facial landmarks even in challenging lighting and head rotations.
2. **Alignment & Crop**:
   Using the 5 facial landmarks, the face is aligned horizontally to remove tilt and cropped to a uniform resolution.
3. **Deep Feature Extraction (SFace)**:
   SFace maps the aligned face into a compact 128-dimensional hypersphere embedding where faces belonging to the same individual cluster together.
4. **Cosine Similarity Matching**:
   The extracted vector is matched against pre-computed embeddings using cosine similarity. If the score meets the confidence threshold (default: `0.38`), the identity is verified.
5. **Anti-Duplicate Attendance Logger**:
   When a recognized face appears, a verification record is written to `logs/attendance.csv`. A configurable cooldown timer (default: 60s) prevents flooding the log file.

---

## ✨ Key Features

* **Real-Time Live Recognition**: Interactive video preview with futuristic corner brackets, landmarks, identity tags, and match confidence percentages.
* **Multi-Shot Face Enrollment**:
  * **Live Capture**: Automatically takes 5 progressive camera frames to compute an averaged facial signature that accommodates varying angles and expressions.
  * **File Upload**: Enroll any person by selecting one or more photos from your disk.
* **Image Inspector**: Analyze any static image file (`.jpg`, `.png`, `.webp`) and export annotated images.
* **Database Management**: View all enrolled identities, count photos, re-index embeddings, or delete entries with one click.
* **Automated Attendance Tracking**: Keeps a timestamped log (`Date, Time, Name, Confidence, Status`) with a one-click button to open directly in Microsoft Excel or spreadsheet software.
* **Real-Time Telemetry & Tuning**: Live FPS counter, active faces count, and interactive sliders for matching threshold and detector sensitivity.
* **Fully English Interface**: All UI text, labels, and messages are standardized in English for global usability.

---

## 📦 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/Abdiaziiz09/Face-detection.git
cd Face-detection
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

> **Note**: The core dependencies are standard wheels:
> - `opencv-python` (Computer Vision & Deep Neural Network engine)
> - `numpy` (Numerical matrix math)
> - `pillow` (Image handling for GUI)
> - `requests` (Model download)
> - `tqdm` (Progress bar)

### Step 4: Download AI Models ⬅️ Required on First Run
The ONNX model files are **not included** in the repository (they are binary files ~39 MB total).  
Run the dedicated setup script **once** after cloning:

```bash
python download_models.py
```

**Expected output:**
```
=======================================================
  FaceEngine AI — Model Setup
=======================================================

[Downloading] YuNet Face Detector
  Size: 232,589 bytes
  [████████████████████] 100.0%
  [OK] 232,589 bytes written.

[Downloading] SFace Face Recognizer
  Size: 38,696,353 bytes
  [████████████████████] 100.0%
  [OK] 38,696,353 bytes written.

=======================================================
  Verifying models with OpenCV ...
=======================================================
  YuNet detector  : OK
  SFace recognizer: OK

  All models are ready — you can now run the app!
```

> **Why aren't models committed to Git?**  
> Large binary files bloat repository history. The models are downloaded directly from the official [OpenCV Zoo](https://github.com/opencv/opencv_zoo) repository, which is the authoritative and trusted source.

---

## 🚀 How to Use

### 1. Modern Desktop GUI (Recommended)

#### On Windows:
Simply double-click the included batch launcher:
```
run_app.bat
```

#### On Any Platform (Terminal):
```bash
python run_gui.py
```

#### Using the Graphical Interface:
1. **Live Camera**: Click **"Start Camera"** to begin real-time face recognition. The system draws green bounding boxes for recognized people and red for unknown individuals.
2. **Enroll Person**:
   - Type the person's name.
   - Click **"Start Live Sampling"** (camera will capture 5 shots) OR click **"Browse Photos from Disk"** to select pictures.
3. **Inspect Image**: Click **"Browse Image"** to analyze any photo on your drive and view full detection metrics.
4. **Identities DB**: View all registered people, remove identities, or open their photo folder.
5. **Attendance Log**: View verified detections and click **"Open in Excel / CSV"** to inspect logs.
6. **Tuning & Sliders**: Adjust matching sensitivity and detection confidence in real time.

---

### 2. Command Line Interface (CLI)

For headless systems, servers, or automated scripts, use `main.py`:

#### Run Real-Time Webcam in CLI:
```bash
python main.py --mode webcam
```
* Press `q` to quit the feed.
* Press `s` to instantly snapshot and enroll a new face.

#### Specify a Different Camera Index:
```bash
python main.py --mode webcam --cam 1
```

#### Analyze a Static Image:
```bash
python main.py --mode image --path "test_image.jpg" --out "output_result.jpg"
```

#### Enroll an Identity from Image Files:
```bash
python main.py --mode enroll --name "Ahmed" --path "photo1.jpg" "photo2.jpg"
```

#### List Enrolled Identities:
```bash
python main.py --mode list
```

#### Rebuild and Synchronize Database:
```bash
python main.py --mode rebuild
```

---

## 📁 Project Directory Structure

```
Face-detection/
├── .gitignore                  # Prevents tracking caches, models, and personal data
├── LICENSE                     # MIT License
├── README.md                   # This documentation file
├── requirements.txt            # Python dependencies (pip install -r requirements.txt)
├── download_models.py          # ⬅️ Run once: downloads & verifies YuNet + SFace models
├── run_app.bat                 # 1-Click Windows GUI launcher
├── run_gui.py                  # Standalone GUI entrypoint
├── main.py                     # Unified CLI & GUI launcher
├── core/
│   ├── __init__.py
│   ├── model_manager.py        # Model path resolution & legacy download helper
│   ├── face_engine.py          # YuNet detector + SFace feature extractor
│   ├── database.py             # Identity registration, storage & embedding cache
│   └── attendance.py           # Attendance CSV logger with cooldown
├── gui/
│   ├── __init__.py
│   └── app.py                  # Polished Dark-Themed Desktop GUI (Tkinter + Pillow)
├── data/
│   ├── known_faces/            # Enrolled photos organized in subfolders by name
│   └── embeddings.pkl          # Cached 128D feature vectors (auto-generated, git-ignored)
├── logs/
│   └── attendance.csv          # Timestamped attendance log (auto-generated, git-ignored)
└── models/
    ├── face_detection_yunet_2023mar.onnx   # Downloaded by download_models.py (git-ignored)
    └── face_recognition_sface_2021dec.onnx # Downloaded by download_models.py (git-ignored)
```

---

## ⚙️ Settings & Fine-Tuning

In the **Tuning & Sliders** tab of the GUI, you can customize:

| Parameter | Default | Range | Description |
| :--- | :---: | :---: | :--- |
| **Match Threshold (Cosine)** | `0.38` | `0.20 – 0.70` | Higher values require a stricter facial match (reduces false positives). Lower values allow more lenient matching across different angles. |
| **Detection Score (YuNet)** | `0.75` | `0.40 – 0.95` | Confidence required to classify a region as a face. Higher values filter out background noise. |
| **Attendance Cooldown** | `60s` | `10s – 300s` | Minimum time before logging the same recognized individual into `attendance.csv` again. |

---

## ❓ Troubleshooting & FAQ

#### 1. "Camera device not found or cannot be opened"
- Check if another application (Zoom, Skype, Teams, Browser) is currently using your webcam.
- If using an external USB webcam, change the camera device index in the GUI dropdown or use `--cam 1`.
- Ensure camera access permissions are enabled in Windows Settings (*Settings > Privacy & Security > Camera*).

#### 2. "Recognition shows Unknown even after enrolling"
- Ensure good lighting when taking enrollment photos. Avoid heavy backlighting.
- Using the **Live Sampling** method in the GUI is recommended because it captures 5 distinct angles to create a robust mean embedding.
- If wearing glasses or face accessories, try slightly lowering the *Match Threshold* in the Settings tab (e.g., from `0.38` to `0.34`).

#### 3. "Does this require GPU / CUDA?"
- No. Both YuNet and SFace are optimized for CPU inference via OpenCV DNN and achieve 30+ FPS on ordinary modern CPUs.

#### 4. "Models missing after cloning from GitHub"
- This is expected. The `.onnx` model files are excluded from Git to keep the repository lightweight.
- Run **`python download_models.py`** once after cloning to download them from OpenCV Zoo (~39 MB total). This only needs to be done once per machine.

#### 5. "Can I use this on a server without a screen (headless)?"
- Yes. Use the CLI interface via `python main.py --mode webcam` or `--mode image`. The GUI (`run_gui.py`) requires a display (Tkinter).

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, distribute, and integrate it into private or commercial projects.

See the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgements

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) — YuNet & SFace open-source ONNX models
- [OpenCV](https://opencv.org/) — Computer vision framework and DNN inference engine
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — Python standard GUI library
