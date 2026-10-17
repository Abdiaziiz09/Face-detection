import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from core.model_manager import ensure_models
from core.face_engine import FaceEngine
from core.database import FaceDatabase
from core.attendance import AttendanceLogger

def run_webcam_cli(engine: FaceEngine, db: FaceDatabase, attendance: AttendanceLogger, cam_idx: int = 0):
    print("=" * 65)
    print("  REAL-TIME WEBCAM FACE RECOGNITION (CLI MODE)")
    print("  Controls:")
    print("   • Press 'q' to exit the camera feed")
    print("   • Press 's' to capture and enroll a new identity")
    print("=" * 65)

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)

    if not cap.isOpened():
        print(f"[Error] Could not open camera device #{cam_idx}.")
        return

    fps = 0.0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, now - last_time))
        last_time = now

        # Run face recognition
        results = engine.process_frame(frame, db.embeddings, draw_annotations=True)

        for r in results:
            if r["is_known"]:
                attendance.log(r["name"], r["confidence_pct"])

        # Display FPS & count on frame
        cv2.putText(frame, f"FPS: {int(fps)} | Faces: {len(results)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Face Recognition (Press 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n[+] Enter person name to register: ", end="", flush=True)
            name = input().strip()
            if name:
                success, msg = db.add_person_from_frames(name, [frame])
                print(f"[Result] {msg}")

    cap.release()
    cv2.destroyAllWindows()

def run_image_cli(engine: FaceEngine, db: FaceDatabase, img_path: str, output_path: str = None):
    p = Path(img_path)
    if not p.exists():
        print(f"[Error] File not found: '{img_path}'.")
        return

    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[Error] Could not decode image: '{img_path}'.")
        return

    results = engine.process_frame(img, db.embeddings, draw_annotations=True)
    print(f"\nAnalysis Report for '{p.name}':")
    print(f"Total faces detected: {len(results)}")
    for idx, r in enumerate(results, start=1):
        status = "VERIFIED" if r["is_known"] else "UNKNOWN"
        print(f" {idx}. Name: {r['name']} | Confidence: {r['confidence_pct']}% | Status: {status} | BBox: {r['box']}")

    out_file = output_path or f"output_{p.name}"
    _, buf = cv2.imencode(p.suffix or ".jpg", img)
    buf.tofile(out_file)
    print(f"[OK] Annotated image saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Antigravity Open-Source Face Recognition System")
    parser.add_argument("--gui", action="store_true", help="Launch Graphical User Interface (Default)")
    parser.add_argument("--mode", choices=["webcam", "image", "enroll", "list", "rebuild"], default="gui")
    parser.add_argument("--name", type=str, help="Person name for enrollment")
    parser.add_argument("--path", type=str, nargs="+", help="Image path(s)")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--out", type=str, help="Output image file path")

    args = parser.parse_args()

    # Launch GUI by default if no CLI options specified
    if args.gui or (len(sys.argv) == 1 and args.mode == "gui"):
        from gui.app import start_gui
        start_gui()
        return

    ensure_models("models")
    engine = FaceEngine()
    db = FaceDatabase(engine=engine)
    attendance = AttendanceLogger()

    if args.mode == "webcam":
        run_webcam_cli(engine, db, attendance, cam_idx=args.cam)
    elif args.mode == "image":
        if not args.path:
            print("[Error] Please specify image path: --path <image.jpg>")
            return
        run_image_cli(engine, db, args.path[0], args.out)
    elif args.mode == "enroll":
        if not args.name:
            print("[Error] Please specify person name: --name 'Person Name'")
            return
        if args.path:
            success, msg = db.add_person_from_images(args.name, args.path)
            print(f"[Result] {msg}")
        else:
            print("[+] Capturing from webcam... Looking for face samples.")
            cap = cv2.VideoCapture(args.cam)
            frames = []
            for _ in range(30):
                ret, f = cap.read()
                if ret:
                    frames.append(f)
                    time.sleep(0.05)
            cap.release()
            success, msg = db.add_person_from_frames(args.name, frames)
            print(f"[Result] {msg}")
    elif args.mode == "list":
        people = db.list_people()
        print(f"\nRegistered Identities ({len(people)}):")
        for p in people:
            print(f" • {p['name']} ({p['photo_count']} photo(s))")
    elif args.mode == "rebuild":
        db.rebuild_all()

if __name__ == "__main__":
    main()
