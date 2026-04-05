"""
Train improved emoji prediction models and save the best to dashboard/models/.

Key improvements over the notebook baseline:
  - class_weight='balanced'  →  stops minority classes (flushed, relaxed) from
    being drowned by the majority class (sob).
  - FeatureUnion of word n-grams (1-2) + character n-grams (3-5)  →  char grams
    capture informal patterns ("lmao", "omg", "haha") without needing tokenisation.

Usage:
    cd <repo-root>
    python src/train.py
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import CUSTOM_STOPS, clean_tweet

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "dashboard" / "models"
RESULTS_DIR = ROOT / "results"


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    tweets = (DATA_DIR / "tweets.txt").read_text().splitlines()
    emojis = (DATA_DIR / "emoji.txt").read_text().splitlines()
    df = pd.DataFrame({
        "text":  [t.strip() for t in tweets],
        "label": [e.strip() for e in emojis],
    })
    df["cleaned"] = df["text"].apply(clean_tweet)
    df = df.drop_duplicates(subset=["cleaned"])
    df = df[df["cleaned"].str.len() > 0]
    print(f"Dataset: {len(df):,} rows, {df['label'].nunique()} classes")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features() -> FeatureUnion:
    """Word (1-2) n-grams + character (3-5) n-grams combined."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
        stop_words=CUSTOM_STOPS,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.95,
        max_features=20_000,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


# ── Model zoo ────────────────────────────────────────────────────────────────

MODELS = {
    "Naive Bayes": Pipeline([
        ("features", build_features()),
        ("clf", MultinomialNB(alpha=0.1)),
    ]),
    "Logistic Regression (balanced)": Pipeline([
        ("features", build_features()),
        ("clf", LogisticRegression(
            C=5.0, class_weight="balanced",
            max_iter=1000, solver="saga",
        )),
    ]),
    "Linear SVM (balanced)": Pipeline([
        ("features", build_features()),
        ("clf", LinearSVC(
            C=1.0, class_weight="balanced",
            max_iter=3000, dual="auto",
        )),
    ]),
}


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    df = load_data()
    X, y = df["cleaned"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    records = []
    best_pipeline, best_f1, best_name = None, 0.0, ""

    for name, pipeline in MODELS.items():
        print(f"\nTraining {name}…")
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc      = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        wt_f1    = f1_score(y_test, preds, average="weighted")

        print(f"  Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}  |  Weighted F1: {wt_f1:.4f}")
        print(classification_report(y_test, preds))

        records.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": macro_f1,
            "f1_weighted": wt_f1,
        })

        if macro_f1 > best_f1:
            best_f1, best_pipeline, best_name = macro_f1, pipeline, name

    # ── Save best model ───────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "tfidf_svm_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"\nBest model: {best_name}  (Macro F1 {best_f1:.4f})")
    print(f"Saved → {model_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    return best_pipeline


if __name__ == "__main__":
    train()
