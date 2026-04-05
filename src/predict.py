"""Prediction logic for TF-IDF SVM and TRIBE v2 pipelines."""
import pickle
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "dashboard" / "models"


def load_svm_pipeline():
    path = MODELS_DIR / "tfidf_svm_pipeline.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_svm(text: str, pipeline=None) -> str:
    if pipeline is None:
        pipeline = load_svm_pipeline()
    return pipeline.predict([text])[0]


def predict_tribe(text: str) -> str:
    # TODO: load TRIBE v2 artifacts from dashboard/models/tribe_v2/
    raise NotImplementedError("TRIBE v2 inference not yet wired up.")
