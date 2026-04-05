"""Prediction helpers for the dashboard."""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "dashboard" / "models"

EMOJI_MAP = {
    "sob": "😭", "heart_eyes": "😍", "weary": "😩",
    "blush": "😊", "wink": "😉", "yum": "😋",
    "smirk": "😏", "grin": "😁", "relaxed": "☺️",
    "flushed": "😳",
}


def load_pipeline(path: Path | None = None):
    path = path or (MODELS_DIR / "tfidf_svm_pipeline.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python src/train.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_top_k(text: str, pipeline, k: int = 3) -> list[tuple[str, float]]:
    """
    Returns a ranked list of (label, score) tuples.
    Uses decision_function scores for LinearSVC or predict_proba for probabilistic models.
    Falls back to top-1 if scores unavailable.
    """
    clf = pipeline.named_steps.get("clf") or pipeline.steps[-1][1]

    if hasattr(clf, "decision_function"):
        scores = pipeline.decision_function([text])[0]
        classes = clf.classes_
    elif hasattr(clf, "predict_proba"):
        scores = pipeline.predict_proba([text])[0]
        classes = clf.classes_
    else:
        pred = pipeline.predict([text])[0]
        return [(pred, 1.0)]

    top_idx = np.argsort(scores)[-k:][::-1]
    return [(classes[i], float(scores[i])) for i in top_idx]
