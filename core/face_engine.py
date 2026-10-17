import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from core.model_manager import ensure_models

class FaceEngine:
    """
    Open-source, free, offline Face Detection & Recognition Engine
    using OpenCV YuNet (Face Detection) + SFace (Face Recognition).
    """

    def __init__(
        self,
        models_dir: str = "models",
        detector_score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        cosine_threshold: float = 0.38,
        l2_threshold: float = 1.128,
    ):
        model_paths = ensure_models(models_dir)
        self.yunet_path = model_paths["yunet"]
        self.sface_path = model_paths["sface"]

        self.score_thresh = detector_score_threshold
        self.nms_thresh = nms_threshold
        self.cosine_thresh = cosine_threshold
        self.l2_thresh = l2_threshold

        # Initialize detector with default resolution (adapts dynamically per frame)
        self.current_input_size = (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=self.current_input_size,
            score_threshold=self.score_thresh,
            nms_threshold=self.nms_thresh,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def set_cosine_threshold(self, threshold: float):
        """Update match cosine threshold."""
        self.cosine_thresh = float(threshold)

    def set_detector_threshold(self, threshold: float):
        """Update detection confidence threshold."""
        self.score_thresh = float(threshold)
        self.detector.setScoreThreshold(self.score_thresh)

    def _update_detector_size(self, width: int, height: int):
        if (width, height) != self.current_input_size:
            self.detector.setInputSize((width, height))
            self.current_input_size = (width, height)

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces in a BGR image.
        Returns: numpy array of faces or None.
        Each face row: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        h, w = image.shape[:2]
        self._update_detector_size(w, h)
        _, faces = self.detector.detect(image)
        return faces

    def extract_embedding(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """
        Aligns, crops face using facial landmarks, and computes the 128D SFace embedding.
        """
        aligned_face = self.recognizer.alignCrop(image, face)
        embedding = self.recognizer.feature(aligned_face)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def compare_embeddings(self, feat1: np.ndarray, feat2: np.ndarray) -> Tuple[float, bool]:
        """
        Compares two face embeddings using Cosine Similarity.
        Returns: (similarity_score, is_match)
        """
        cosine_score = float(self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
        is_match = cosine_score >= self.cosine_thresh
        return cosine_score, is_match

    def process_frame(
        self,
        frame: np.ndarray,
        known_embeddings: Dict[str, np.ndarray],
        draw_annotations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Processes an image frame:
        1. Detects all faces
        2. Extracts embeddings
        3. Identifies known identities
        4. Optionally draws clean annotations
        """
        faces = self.detect_faces(frame)
        results = []

        if faces is None or len(faces) == 0:
            return results

        h, w = frame.shape[:2]

        for face in faces:
            bbox = face[0:4].astype(int)
            x, y, fw, fh = bbox
            score = float(face[14])
            landmarks = face[4:14].reshape((5, 2)).astype(int)

            # Keep bounding box within frame
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            if fw < 10 or fh < 10:
                continue

            feat = self.extract_embedding(frame, face)

            best_name = "Unknown"
            best_score = -1.0

            # Match against database of registered faces
            for name, known_feat in known_embeddings.items():
                cosine_sim, is_match = self.compare_embeddings(feat, known_feat)
                if cosine_sim > best_score:
                    best_score = cosine_sim
                    if is_match:
                        best_name = name

            # Convert cosine score to a calibrated 0-100% confidence value
            if best_score < 0:
                match_pct = 0.0
            else:
                match_pct = min(100.0, max(0.0, (best_score / 0.82) * 100.0))

            face_info = {
                "box": (x, y, fw, fh),
                "landmarks": landmarks,
                "detection_score": score,
                "name": best_name,
                "similarity": best_score,
                "confidence_pct": round(match_pct, 1),
                "is_known": (best_name != "Unknown"),
                "embedding": feat
            }
            results.append(face_info)

            if draw_annotations:
                self.draw_face(frame, face_info)

        return results

    def draw_face(self, frame: np.ndarray, info: Dict[str, Any]):
        """
        Draws sleek futuristic bounding boxes, badge labels, and facial landmarks.
        """
        x, y, w, h = info["box"]
        is_known = info["is_known"]
        name = info["name"]
        pct = info["confidence_pct"]

        # Green for recognized, Soft Red/Coral for Unknown
        theme_color = (46, 213, 115) if is_known else (71, 71, 255)
        corner_len = min(20, w // 4, h // 4)
        thickness = 2

        # 4 Corner Accents
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), theme_color, thickness + 1)
        cv2.line(frame, (x, y), (x, y + corner_len), theme_color, thickness + 1)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), theme_color, thickness + 1)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), theme_color, thickness + 1)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), theme_color, thickness + 1)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), theme_color, thickness + 1)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), theme_color, thickness + 1)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), theme_color, thickness + 1)

        # Outer subtle rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), theme_color, 1)

        # Facial landmarks (eyes, nose, mouth)
        landmark_colors = [
            (255, 200, 0),   # right eye
            (0, 200, 255),   # left eye
            (0, 255, 128),   # nose tip
            (255, 100, 255), # right mouth corner
            (100, 255, 255)  # left mouth corner
        ]
        for idx, pt in enumerate(info["landmarks"]):
            color = landmark_colors[idx % len(landmark_colors)]
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, -1)

        # Text label
        label = f"{name} ({pct:.0f}%)" if is_known else "Unknown"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        font_thickness = 1
        (txt_w, txt_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        lbl_y1 = max(0, y - txt_h - 10)
        lbl_y2 = y
        cv2.rectangle(frame, (x, lbl_y1), (x + txt_w + 14, lbl_y2), theme_color, -1)
        cv2.putText(
            frame,
            label,
            (x + 7, y - 6),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
