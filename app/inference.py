"""Inference engine and the post-processing rules.

The same `Engine.predict` path serves `/predict`, `/predict/batch`, and batch jobs.
The classical branch always runs. An encoder is used only after `attach_encoder`;
if that branch raises, that ticket keeps the classical probabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.fusion import apply_fusion
from app.labels import NEVER_URGENT, SECONDARY_CATEGORIES, TEAM_BY_CATEGORY

logger = logging.getLogger("tensorforge")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DEFAULT_ARTIFACT = ARTIFACTS / "classical.joblib"
MANIFEST_PATH = ARTIFACTS / "manifest.json"


def align_columns(proba: np.ndarray, classes: list[Any], labels: list[str]) -> np.ndarray:
    """Reorder `predict_proba` columns to `labels`. Missing classes stay 0."""
    proba = np.asarray(proba, dtype=np.float64)
    index = {str(classes[i]): i for i in range(len(classes))}
    out = np.zeros((proba.shape[0], len(labels)), dtype=np.float64)
    for column, label in enumerate(labels):
        source = index.get(str(label))
        if source is not None:
            out[:, column] = proba[:, source]
    return out


def positive_proba(model: Any, matrix: Any) -> np.ndarray:
    """P(class == 1) from a binary estimator. Works for bool and int labels."""
    proba = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    for column, label in enumerate(model.classes_):
        if label == 1:
            return proba[:, column]
    return np.zeros(proba.shape[0], dtype=np.float64)


def apply_rules(
    category_proba: np.ndarray,
    category_labels: list[str],
    secondary_proba: np.ndarray,
    secondary_labels: list[str],
    urgent_proba: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Turn one ticket's probabilities into a contract-safe prediction.

    Category is argmax. Secondary is the best of the five observed classes other
    than the primary, kept only when it clears the threshold. Urgency is forced
    off for the four categories that are never urgent, and spam also clears secondary.
    """
    category_index = int(np.argmax(category_proba))
    category = category_labels[category_index]
    confidence = float(np.clip(category_proba[category_index], 0.0, 1.0))
    confidence = float(round(confidence, 4))

    secondary: str | None = None
    if len(secondary_labels):
        candidates = [
            (float(secondary_proba[i]), secondary_labels[i])
            for i in range(len(secondary_labels))
            if secondary_labels[i] != category
        ]
        if candidates:
            best_p, best_label = max(candidates)
            if best_p >= thresholds["secondary"]:
                secondary = best_label

    is_urgent = float(urgent_proba) >= thresholds["urgent"]
    if category in NEVER_URGENT:
        is_urgent = False
    if category == "spam_irrelevant":
        secondary = None
        is_urgent = False
    if secondary == category:
        secondary = None

    return {
        "category": category,
        "secondary_category": secondary,
        "team": TEAM_BY_CATEGORY[category],
        "is_urgent": bool(is_urgent),
        "confidence": confidence,
        "needs_human_review": confidence < thresholds["review"],
    }


def resolve_field_priority(
    combined: dict[str, Any],
    body: dict[str, Any] | None,
    subject: dict[str, Any] | None,
    *,
    body_is_vague: bool,
    subject_is_vague: bool,
) -> dict[str, Any]:
    """Choose a prediction when the subject and the body can name different issues.

    A missing or vague subject keeps the combined score. Otherwise:

    1. A safety call on a field that actually names an issue wins, from either side.
    2. A vague body ("please help", "see below") follows the subject.
    3. If the two categories differ, the body wins. Review stays on that confidence.
    4. If they agree, the combined score stands. That is the input the model was trained on.
    """
    if body is None or subject is None or subject_is_vague:
        return combined

    body_safety = body["category"] == "safety_conduct" and not body_is_vague
    subject_safety = subject["category"] == "safety_conduct"
    if body_safety:
        return body
    if subject_safety:
        return subject
    if body_is_vague:
        return subject
    if body["category"] != subject["category"]:
        return body
    return combined


def fuse_ticket_probabilities(
    category_proba: np.ndarray,
    secondary_proba: np.ndarray,
    urgent_proba: np.ndarray,
    texts: list[str],
    encoder: Any | None,
    fusion: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mix in the encoder one ticket at a time. A failure keeps that classical row."""
    if encoder is None or fusion is None:
        return category_proba, secondary_proba, urgent_proba
    category = np.array(category_proba, dtype=np.float64, copy=True)
    secondary = np.array(secondary_proba, dtype=np.float64, copy=True)
    urgent = np.array(urgent_proba, dtype=np.float64, copy=True)
    for index, text in enumerate(texts):
        fused = _fuse_one(
            category_proba[index : index + 1],
            secondary_proba[index : index + 1],
            urgent_proba[index : index + 1],
            text,
            encoder,
            fusion,
        )
        if fused is None:
            continue
        category[index] = fused[0][0]
        secondary[index] = fused[1][0]
        urgent[index] = fused[2][0]
    return category, secondary, urgent


def _fuse_one(
    category_proba: np.ndarray,
    secondary_proba: np.ndarray,
    urgent_proba: np.ndarray,
    text: str,
    encoder: Any,
    fusion: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        enc_cat, enc_sec, enc_urg = encoder.probabilities([text])
        enc_cat = np.asarray(enc_cat, dtype=np.float64)
        enc_sec = np.asarray(enc_sec, dtype=np.float64)
        enc_urg = np.asarray(enc_urg, dtype=np.float64).reshape(-1)
        if enc_cat.ndim == 1:
            enc_cat = enc_cat.reshape(1, -1)
        if enc_sec.ndim == 1:
            enc_sec = enc_sec.reshape(1, -1)
        if enc_cat.shape != (1, category_proba.shape[1]) or enc_sec.shape != (
            1,
            secondary_proba.shape[1],
        ):
            raise RuntimeError(
                f"encoder probability shape {enc_cat.shape} / {enc_sec.shape} does not match the classical head"
            )
        if enc_urg.shape != (1,):
            raise RuntimeError(f"encoder urgent shape {enc_urg.shape}")
        return apply_fusion(
            {
                "category_proba": category_proba,
                "secondary_proba": secondary_proba,
                "urgent_proba": urgent_proba,
            },
            {
                "category_proba": enc_cat,
                "secondary_proba": enc_sec,
                "urgent_proba": enc_urg,
            },
            fusion,
        )
    except Exception:
        logger.exception("encoder failed on one ticket; using the classical branch")
        return None


class Engine:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.model_version: str = str(bundle["model_version"])
        self.config_name: str = str(bundle.get("config_name", ""))
        self.vectorizer = bundle["vectorizer"]
        self.category_model = bundle["category_model"]
        self.category_labels: list[str] = list(bundle["category_labels"])
        self.secondary_models: dict[str, Any] = dict(bundle["secondary_models"])
        self.secondary_labels: list[str] = list(
            bundle.get("secondary_labels", SECONDARY_CATEGORIES)
        )
        self.urgent_model = bundle["urgent_model"]
        self.thresholds: dict[str, float] = {
            "secondary": float(bundle["thresholds"]["secondary"]),
            "urgent": float(bundle["thresholds"]["urgent"]),
            "review": float(bundle["thresholds"]["review"]),
        }
        self.encoder: Any | None = None
        self.fusion: dict[str, Any] | None = None

    def attach_encoder(self, encoder: Any, fusion: dict[str, Any]) -> None:
        """Turn on fusion. `fusion` is the settings object written by `ml.fuse`."""
        self.encoder = encoder
        self.fusion = fusion

    @classmethod
    def load(cls, path: Path | None = None) -> Engine:
        """Load the classical bundle. No path and a manifest means the frozen version."""
        if path is None and MANIFEST_PATH.is_file():
            return cls.load_manifest(MANIFEST_PATH)
        return cls._load_joblib(path or DEFAULT_ARTIFACT)

    @classmethod
    def load_manifest(cls, path: Path) -> Engine:
        from app.manifest import verify

        manifest = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent
        verify(manifest, root)
        if "classical.joblib" not in manifest["files"]:
            raise RuntimeError("manifest is missing classical.joblib")
        engine = cls._load_joblib(root / "classical.joblib")
        engine.model_version = str(manifest["model_version"])
        if manifest.get("encoder_enabled"):
            engine._attach_from_manifest(root, manifest)
        return engine

    def _attach_from_manifest(self, root: Path, manifest: dict[str, Any]) -> None:
        from app.onnx_encoder import OnnxEncoder

        fusion_path = root / "fusion.json"
        meta_path = root / "encoder_meta.json"
        fusion_payload = json.loads(fusion_path.read_text(encoding="utf-8"))
        settings = fusion_payload.get("settings", fusion_payload)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        encoder = OnnxEncoder.load(root / "encoder.int8.onnx", root / "tokenizer.json", meta)
        self.attach_encoder(encoder, settings)
        thresholds = settings.get("thresholds")
        if isinstance(thresholds, dict):
            self.thresholds = {
                "secondary": float(thresholds["secondary"]),
                "urgent": float(thresholds["urgent"]),
                "review": float(thresholds["review"]),
            }
        if "model_version" in manifest:
            self.model_version = str(manifest["model_version"])

    @classmethod
    def _load_joblib(cls, path: Path) -> Engine:
        import joblib

        # Trusted artifact we trained and hashed at freeze time (S301 is our own joblib).
        bundle = joblib.load(path)  # noqa: S301
        return cls(bundle)

    def predict(self, ticket: dict[str, Any]) -> dict[str, Any]:
        return self.predict_many([ticket])[0]

    def _score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        """Probabilities and rules for each already-built model string."""
        matrix = self.vectorizer.transform(texts)
        category_proba = align_columns(
            self.category_model.predict_proba(matrix),
            list(self.category_model.classes_),
            self.category_labels,
        )
        secondary_columns = []
        for label in self.secondary_labels:
            model = self.secondary_models.get(label)
            if model is None:
                secondary_columns.append(np.zeros(len(texts), dtype=np.float64))
            else:
                secondary_columns.append(positive_proba(model, matrix))
        secondary_proba = np.column_stack(secondary_columns)
        urgent = positive_proba(self.urgent_model, matrix)
        category_proba, secondary_proba, urgent = fuse_ticket_probabilities(
            category_proba,
            secondary_proba,
            urgent,
            texts,
            self.encoder,
            self.fusion,
        )
        return [
            apply_rules(
                category_proba[row],
                self.category_labels,
                secondary_proba[row],
                self.secondary_labels,
                float(urgent[row]),
                self.thresholds,
            )
            for row in range(len(texts))
        ]

    def predict_many(self, tickets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from app.text import build_input, is_vague_body, normalize

        if not tickets:
            return []
        texts: list[str] = []
        # combined index, then body and subject indexes when the subject names an issue
        layout: list[tuple[int, int | None, int | None]] = []
        for ticket in tickets:
            channel = ticket["channel"]
            subject = ticket.get("subject") or ""
            text = ticket["text"]
            combined_index = len(texts)
            texts.append(build_input(channel, subject, text))
            body_index = None
            subject_index = None
            # A vague subject ("Re:", "hello") is not a second issue. Score the ticket once.
            if normalize(subject) and not is_vague_body(subject):
                body_index = len(texts)
                texts.append(build_input(channel, None, text))
                subject_index = len(texts)
                texts.append(build_input(channel, None, subject))
            layout.append((combined_index, body_index, subject_index))

        scored = self._score_texts(texts)
        predictions: list[dict[str, Any]] = []
        for ticket, (combined_index, body_index, subject_index) in zip(
            tickets, layout, strict=True
        ):
            body_pred = scored[body_index] if body_index is not None else None
            subject_pred = scored[subject_index] if subject_index is not None else None
            chosen = resolve_field_priority(
                scored[combined_index],
                body_pred,
                subject_pred,
                body_is_vague=is_vague_body(ticket["text"]),
                subject_is_vague=is_vague_body(ticket.get("subject") or ""),
            )
            pred = dict(chosen)
            pred["model_version"] = self.model_version
            if ticket.get("ticket_id") is not None:
                pred["ticket_id"] = ticket["ticket_id"]
            predictions.append(pred)
        return predictions
