import os
import pickle
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

class FaceDatabase:
    """
    Manages registered identities, photo datasets, and cached face embeddings.
    """

    def __init__(self, data_dir: str = "data", engine = None):
        self.data_dir = Path(data_dir).resolve()
        self.faces_dir = self.data_dir / "known_faces"
        self.cache_file = self.data_dir / "embeddings.pkl"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        self.engine = engine
        self.embeddings: Dict[str, np.ndarray] = {}
        self.multi_embeddings: Dict[str, List[np.ndarray]] = {}

        self.load()

    def set_engine(self, engine):
        self.engine = engine

    def load(self):
        """Loads cached embeddings from disk if available."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.embeddings = data.get("embeddings", {})
                    self.multi_embeddings = data.get("multi_embeddings", {})
                print(f"[Database] Loaded {len(self.embeddings)} registered identity/identities from cache.")
                return
            except Exception as e:
                print(f"[Database Warning] Failed to load cache: {e}. Rebuilding...")

        if self.engine is not None:
            self.rebuild_all()

    def save(self):
        """Saves current embeddings to disk cache."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump({
                    "embeddings": self.embeddings,
                    "multi_embeddings": self.multi_embeddings
                }, f)
        except Exception as e:
            print(f"[Database Error] Could not save cache: {e}")

    def list_people(self) -> List[Dict[str, Any]]:
        """Returns list of registered people with photo count and status."""
        people = []
        for person_dir in self.faces_dir.iterdir():
            if person_dir.is_dir():
                name = person_dir.name
                images = [p for p in person_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
                has_embedding = name in self.embeddings
                people.append({
                    "name": name,
                    "photo_count": len(images),
                    "is_active": has_embedding,
                    "folder": str(person_dir)
                })
        return sorted(people, key=lambda x: x["name"].lower())

    def add_person_from_images(self, person_name: str, image_paths: List[str]) -> Tuple[bool, str]:
        """Enrolls a person using a list of image file paths."""
        if not self.engine:
            return False, "Face engine is not initialized."

        cleaned_name = person_name.strip()
        if not cleaned_name:
            return False, "Person name cannot be empty."

        person_dir = self.faces_dir / cleaned_name
        person_dir.mkdir(parents=True, exist_ok=True)

        embeddings_list = []
        saved_count = 0

        for idx, img_path in enumerate(image_paths):
            path_obj = Path(img_path)
            if not path_obj.exists():
                continue

            img = cv2.imdecode(np.fromfile(str(path_obj), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            faces = self.engine.detect_faces(img)
            if faces is None or len(faces) == 0:
                continue

            best_face = max(faces, key=lambda f: f[2] * f[3])
            emb = self.engine.extract_embedding(img, best_face)
            embeddings_list.append(emb)

            dest_file = person_dir / f"sample_{idx+1}_{path_obj.name}"
            try:
                shutil.copy2(path_obj, dest_file)
                saved_count += 1
            except Exception:
                pass

        if not embeddings_list:
            return False, f"No clear face detected in the selected photos for: {cleaned_name}"

        # Compute normalized mean embedding
        mean_emb = np.mean(embeddings_list, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        self.embeddings[cleaned_name] = mean_emb
        self.multi_embeddings[cleaned_name] = embeddings_list
        self.save()

        return True, f"Successfully enrolled '{cleaned_name}' using {len(embeddings_list)} photo(s)!"

    def add_person_from_frames(self, person_name: str, frames: List[np.ndarray]) -> Tuple[bool, str]:
        """Enrolls a person directly from captured video/webcam frames."""
        if not self.engine:
            return False, "Face engine is not initialized."

        cleaned_name = person_name.strip()
        if not cleaned_name:
            return False, "Person name cannot be empty."

        person_dir = self.faces_dir / cleaned_name
        person_dir.mkdir(parents=True, exist_ok=True)

        embeddings_list = []
        for idx, frame in enumerate(frames):
            faces = self.engine.detect_faces(frame)
            if faces is None or len(faces) == 0:
                continue

            best_face = max(faces, key=lambda f: f[2] * f[3])
            emb = self.engine.extract_embedding(frame, best_face)
            embeddings_list.append(emb)

            crop_path = person_dir / f"capture_{idx+1}.jpg"
            _, buf = cv2.imencode(".jpg", frame)
            buf.tofile(str(crop_path))

        if not embeddings_list:
            return False, "No face was captured from the camera. Please try again."

        mean_emb = np.mean(embeddings_list, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        self.embeddings[cleaned_name] = mean_emb
        self.multi_embeddings[cleaned_name] = embeddings_list
        self.save()

        return True, f"Successfully enrolled '{cleaned_name}' with {len(embeddings_list)} sample(s)."

    def delete_person(self, person_name: str) -> bool:
        """Deletes a person from the database and removes their files."""
        self.embeddings.pop(person_name, None)
        self.multi_embeddings.pop(person_name, None)
        self.save()

        person_dir = self.faces_dir / person_name
        if person_dir.exists() and person_dir.is_dir():
            shutil.rmtree(person_dir, ignore_errors=True)
            return True
        return False

    def rebuild_all(self):
        """Scans data/known_faces and recomputes all embeddings."""
        if not self.engine:
            return

        print("[Database] Re-indexing all registered faces...")
        new_embeddings = {}
        new_multi = {}

        for person_dir in self.faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            img_files = [p for p in person_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
            
            embs = []
            for img_path in img_files:
                img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                faces = self.engine.detect_faces(img)
                if faces is None or len(faces) == 0:
                    continue
                best_face = max(faces, key=lambda f: f[2] * f[3])
                emb = self.engine.extract_embedding(img, best_face)
                embs.append(emb)

            if embs:
                mean_emb = np.mean(embs, axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
                new_embeddings[name] = mean_emb
                new_multi[name] = embs
                print(f" -> Enrolled identity: '{name}' ({len(embs)} image(s))")

        self.embeddings = new_embeddings
        self.multi_embeddings = new_multi
        self.save()
        print(f"[Database] Complete! {len(self.embeddings)} identities ready.")
