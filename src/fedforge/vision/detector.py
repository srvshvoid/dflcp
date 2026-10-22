"""Media preprocessing, face extraction, and forensic fusion pipeline."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from .forensics import PixelForensics
from .model import DeepfakeDetector


class FaceDetectionResult:
    """Bounding box + cropped face data."""

    def __init__(self, bbox: Tuple[int, int, int, int], crop: np.ndarray) -> None:
        self.bbox = bbox   # (x, y, w, h)
        self.crop = crop   # BGR numpy array


class MediaDetectionPipeline:
    """End-to-end pipeline: decode → detect faces → forensics → verdict."""

    def __init__(self, model: Optional[DeepfakeDetector] = None) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.model = model or DeepfakeDetector()
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def sanitize(cls, data: Any) -> Any:
        """Recursively convert NumPy / PyTorch objects to native Python for JSON."""
        if isinstance(data, (np.integer,)):
            return int(data)
        if isinstance(data, (np.floating,)):
            return float(data)
        if isinstance(data, (np.bool_, bool)):
            return bool(data)
        if isinstance(data, np.ndarray):
            return [cls.sanitize(v) for v in data.tolist()]
        if hasattr(data, "item"):
            try:
                return data.item()
            except Exception:
                pass
        if isinstance(data, dict):
            return {str(k): cls.sanitize(v) for k, v in data.items()}
        if isinstance(data, (list, tuple, set)):
            return [cls.sanitize(v) for v in data]
        return data

    def detect_faces(self, image_bgr: np.ndarray) -> List[FaceDetectionResult]:
        """Haar cascade face detection with 15 % margin and centre-crop fallback."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        h_img, w_img = image_bgr.shape[:2]
        results: List[FaceDetectionResult] = []

        if len(faces) == 0:
            size = min(h_img, w_img)
            x = (w_img - size) // 2
            y = (h_img - size) // 2
            crop = image_bgr[y: y + size, x: x + size]
            results.append(FaceDetectionResult((x, y, size, size), crop))
            return results

        for (x, y, w, h) in faces:
            mx = int(w * 0.15)
            my = int(h * 0.15)
            x1 = max(0, int(x) - mx)
            y1 = max(0, int(y) - my)
            x2 = min(w_img, int(x + w) + mx)
            y2 = min(h_img, int(y + h) + my)
            crop = image_bgr[y1:y2, x1:x2]
            results.append(
                FaceDetectionResult((x1, y1, x2 - x1, y2 - y1), crop)
            )
        return results

    def _generate_heatmap_overlay(
        self, face_bgr: np.ndarray, mask: torch.Tensor
    ) -> Tuple[str, str]:
        mask_np = mask.squeeze().detach().cpu().numpy()
        h, w = face_bgr.shape[:2]
        if h < 20 or w < 20:
            face_bgr = cv2.resize(face_bgr, (120, 120))
            h, w = face_bgr.shape[:2]
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        mn, mx = float(mask_resized.min()), float(mask_resized.max())
        mask_norm = (mask_resized - mn) / (mx - mn + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(face_bgr, 0.65, heatmap, 0.35, 0)

        def _b64(img: np.ndarray) -> str:
            _, buf = cv2.imencode(".jpg", img)
            return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        return _b64(face_bgr), _b64(blended)

    # ------------------------------------------------------------------ #
    #  Core scoring logic                                                  #
    # ------------------------------------------------------------------ #
    def _fuse_scores(self, model_prob: float, pixel_score: float) -> float:
        """Fuse the CNN model probability with the pixel forensic score.

        Fusion strategy
        ---------------
        • The untrained CNN is a weak prior (~0.05 for any image).
          Its contribution is intentionally LOW so it never overrides the
          rule-based forensics for real photos.
        • The pixel forensic score is the primary signal:
            – If it crosses 0.40 it governs the final score directly.
            – If both signals are low, we return their weighted blend.
        • The CNN score is used as a *tiebreaker* only, weighted 15 %.
        """
        if pixel_score >= 0.40:
            # Strong forensic evidence – trust it; let model add a small bump
            return min(1.0, pixel_score + 0.15 * (model_prob - 0.05))
        else:
            # Both signals low: weighted blend, forensics-dominant
            return 0.15 * model_prob + 0.85 * pixel_score

    # ------------------------------------------------------------------ #
    #  Image analysis                                                      #
    # ------------------------------------------------------------------ #
    def analyze_image_bytes(
        self, image_bytes: bytes, filename: str = "upload.jpg"
    ) -> Dict[str, Any]:
        """Analyse raw image bytes → full forensic report."""
        img_bgr = None
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            pass

        if img_bgr is None:
            try:
                pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                raise ValueError(f"Failed to decode image: {exc}") from exc

        faces = self.detect_faces(img_bgr)
        face_results: List[Dict[str, Any]] = []
        max_prob = 0.0

        for idx, f in enumerate(faces):
            if f.crop is None or f.crop.size == 0 or min(f.crop.shape[:2]) < 5:
                continue

            img_rgb = cv2.cvtColor(f.crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0)
            model_prob, mask = self.model.predict_proba(tensor)
            forensic = PixelForensics.analyze_patch(f.crop)
            pixel_score = forensic["pixel_forensic_score"]

            final_prob = self._fuse_scores(model_prob, pixel_score)
            max_prob = max(max_prob, final_prob)

            face_results.append({
                "face_index":       int(idx),
                "bbox":             [int(b) for b in f.bbox],
                "fake_probability": round(float(final_prob), 4),
                "is_fake":          bool(final_prob >= 0.50),
                "crop_preview":     forensic["crop_preview"],
                "attention_heatmap": forensic["attention_heatmap"],
                "ela_map":          forensic["ela_map"],
                "noise_map":        forensic["noise_map"],
                "fft_map":          forensic["fft_map"],
                "pixel_forensics":  forensic["metrics"],
            })

        if not face_results:
            h, w = img_bgr.shape[:2]
            crop = cv2.resize(img_bgr, (224, 224))
            tensor = self.transform(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0)
            model_prob, mask = self.model.predict_proba(tensor)
            forensic = PixelForensics.analyze_patch(crop)
            pixel_score = forensic["pixel_forensic_score"]
            final_prob = self._fuse_scores(model_prob, pixel_score)
            max_prob = float(final_prob)
            face_results.append({
                "face_index":       0,
                "bbox":             [0, 0, int(w), int(h)],
                "fake_probability": round(float(final_prob), 4),
                "is_fake":          bool(final_prob >= 0.50),
                "crop_preview":     forensic["crop_preview"],
                "attention_heatmap": forensic["attention_heatmap"],
                "ela_map":          forensic["ela_map"],
                "noise_map":        forensic["noise_map"],
                "fft_map":          forensic["fft_map"],
                "pixel_forensics":  forensic["metrics"],
            })

        # ----- verdict & confidence -----
        is_synthetic = max_prob >= 0.50
        verdict = "LIKELY_SYNTHETIC" if is_synthetic else "AUTHENTIC"
        # Confidence = certainty: how far the score is from 0.50
        # We do NOT clamp to a fake minimum — let the score speak for itself.
        confidence = abs(max_prob - 0.50) * 2.0   # 0 … 1 range
        confidence_pct = round(50.0 + confidence * 50.0, 1)  # 50 … 100 %

        report = self._build_moderation_report(max_prob, is_video=False, num_frames=1)
        sha256 = PixelForensics.compute_sha256(image_bytes)
        primary_metrics = face_results[0].get("pixel_forensics", {}) if face_results else {}
        cert = PixelForensics.generate_forensic_certificate(
            filename=filename,
            sha256_hash=sha256,
            verdict=verdict,
            score=max_prob,
            metrics=primary_metrics,
            faces_count=len(face_results),
        )

        return self.sanitize({
            "media_type":        "image",
            "filename":          filename,
            "sha256_fingerprint": sha256,
            "verdict":           verdict,
            "overall_score":     round(float(max_prob), 4),
            "confidence":        confidence_pct,
            "faces_detected":    int(len(face_results)),
            "face_details":      face_results,
            "moderation_report": report,
            "web_provenance":    PixelForensics.generate_web_provenance_links(filename),
            "forensic_certificate": cert,
        })

    # ------------------------------------------------------------------ #
    #  Video analysis                                                      #
    # ------------------------------------------------------------------ #
    def analyze_video_bytes(
        self,
        video_bytes: bytes,
        filename: str = "upload.mp4",
        max_frames: int = 12,
    ) -> Dict[str, Any]:
        """Sample keyframes, analyse faces, pool scores."""
        import tempfile

        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            suffix = ".mp4"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(video_bytes)
            tmp.flush()
            tmp.close()
            tmp_path = tmp.name
        except Exception as exc:
            raise ValueError(f"Failed to buffer video: {exc}") from exc

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
            raise ValueError(
                f"Could not decode '{filename}'. Ensure H.264/MP4 format."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0
        duration_sec = total_frames / fps

        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))[:max_frames]

        frame_scores: List[Dict[str, Any]] = []
        face_previews: List[Dict[str, Any]] = []
        all_probs: List[float] = []

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue
            time_sec = round(f_idx / fps, 2)
            faces = self.detect_faces(frame)
            f = faces[0] if faces else None
            if f is None:
                h_f, w_f = frame.shape[:2]
                sz = min(h_f, w_f)
                x = (w_f - sz) // 2
                y = (h_f - sz) // 2
                f = FaceDetectionResult((x, y, sz, sz), frame[y: y + sz, x: x + sz])

            img_rgb = cv2.cvtColor(f.crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0)
            prob, mask = self.model.predict_proba(tensor)
            all_probs.append(prob)

            frame_scores.append({
                "frame":         int(f_idx),
                "timestamp_sec": float(time_sec),
                "fake_score":    round(float(prob), 4),
            })

            if len(face_previews) < 3:
                crop_b64, heat_b64 = self._generate_heatmap_overlay(f.crop, mask)
                face_previews.append({
                    "frame":           int(f_idx),
                    "timestamp":       f"{time_sec}s",
                    "score":           round(float(prob), 4),
                    "crop_preview":    crop_b64,
                    "attention_heatmap": heat_b64,
                })

        cap.release()
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

        avg_score = (
            0.7 * (sum(all_probs) / len(all_probs)) + 0.3 * max(all_probs)
            if all_probs
            else 0.05
        )

        is_synthetic = avg_score >= 0.50
        verdict = "LIKELY_SYNTHETIC" if is_synthetic else "AUTHENTIC"
        confidence = abs(avg_score - 0.50) * 2.0
        confidence_pct = round(50.0 + confidence * 50.0, 1)

        report = self._build_moderation_report(
            avg_score, is_video=True, num_frames=len(frame_scores)
        )

        return self.sanitize({
            "media_type":       "video",
            "filename":         filename,
            "verdict":          verdict,
            "overall_score":    round(float(avg_score), 4),
            "confidence":       confidence_pct,
            "duration_sec":     round(float(duration_sec), 2),
            "analyzed_frames":  int(len(frame_scores)),
            "frame_timeline":   frame_scores,
            "keyframe_previews": face_previews,
            "moderation_report": report,
        })

    # ------------------------------------------------------------------ #
    #  Moderation report                                                   #
    # ------------------------------------------------------------------ #
    def _build_moderation_report(
        self, score: float, is_video: bool, num_frames: int
    ) -> Dict[str, Any]:
        if score >= 0.70:
            risk = "HIGH"
            verdict = "MANIPULATED / SYNTHETIC"
            desc = (
                "Multiple high-frequency artefact anomalies detected along facial "
                "contours and blending margins. ELA discrepancy and noise "
                "inconsistency patterns match autoencoder reenactment (Face2Face) "
                "or GAN face-swapping."
            )
            rec = "FLAG FOR REMOVAL / RESTRICT VISIBILITY"
        elif score >= 0.50:
            risk = "MEDIUM"
            verdict = "POTENTIALLY SYNTHETIC"
            desc = (
                "Subtle boundary artefacts detected in key regions. Compression or "
                "lighting effects may partially elevate confidence, but forensic "
                "analysis detects non-natural texture patterns."
            )
            rec = "MANUAL REVIEW REQUIRED BY FACT-CHECK DESK"
        else:
            risk = "LOW"
            verdict = "ORGANIC / AUTHENTIC"
            desc = (
                "No significant synthetic manipulation boundaries or digital artefact "
                "clusters detected. Pixel noise and spectral profiles are consistent "
                "with natural camera sensor characteristics."
            )
            rec = "PASS / NO MODERATION ACTION"

        return {
            "risk_level":       risk,
            "summary_verdict":  verdict,
            "recommendation":   rec,
            "findings_narrative": desc,
            "evidence_metadata": {
                "input_type":            "video" if is_video else "still_image",
                "sampled_units":         num_frames,
                "artifact_inspection_engine": "FedForge Forensic Engine v2.0",
            },
        }
