"""Probability fusion shared by the offline fit and the serving engine.

Category temperature is applied after the weighted average. Secondary and urgent
temperatures are applied on each branch, then the branches are averaged. That is
the procedure ``ml.fuse`` fits.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def scale_multiclass(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling. Argmax is unchanged for any temperature above 0."""
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def scale_binary(proba: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(proba, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def mix(classical: np.ndarray, encoder: np.ndarray, encoder_weight: float) -> np.ndarray:
    return (1.0 - encoder_weight) * classical + encoder_weight * encoder


def apply_fusion(
    classical: dict[str, np.ndarray],
    encoder: dict[str, np.ndarray],
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = settings["encoder_weight"]
    temperatures = settings["temperature"]
    category = scale_multiclass(
        mix(classical["category_proba"], encoder["category_proba"], weights["category"]),
        temperatures["category"],
    )
    secondary = mix(
        scale_binary(classical["secondary_proba"], temperatures["secondary"]),
        scale_binary(encoder["secondary_proba"], temperatures["secondary"]),
        weights["secondary"],
    )
    urgent = mix(
        scale_binary(classical["urgent_proba"], temperatures["urgent"]),
        scale_binary(encoder["urgent_proba"], temperatures["urgent"]),
        weights["urgent"],
    )
    return category, secondary, urgent
