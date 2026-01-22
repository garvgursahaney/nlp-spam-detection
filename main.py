# NLP Spam Detection by Garv Gursahaney
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception:
    PorterStemmer = None
    WordNetLemmatizer = None

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "spam.csv"

# Load and prepare the dataset
def ensure_dataset(path: Path) -> None:
    if path.exists():
        return

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("SMSSpamCollection") as f:
            raw = f.read().decode("utf-8", errors="replace")

    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        rows.append(parts)

    df_local = pd.DataFrame(rows, columns=["v1", "v2"])
    df_local.to_csv(path, index=False, encoding="utf-8")


STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer() if PorterStemmer else None
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer else None


def load_dataset(path: Path) -> pd.DataFrame:
    ensure_dataset(path)
    df = pd.read_csv(path, usecols=["v1", "v2"])
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def normalize_text(text: str, *, mode: str = "lemma") -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS]

    if mode == "stem" and STEMMER:
        tokens = [STEMMER.stem(w) for w in tokens]
    elif mode == "lemma" and LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(w) for w in tokens]

    return " ".join(tokens)


def build_search_space() -> tuple[dict[str, Pipeline], dict[str, dict]]:
    pipelines = {
        "tfidf_lr": Pipeline(
            [("vec", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=2000))]
        ),
        "bow_lr": Pipeline(
            [("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=2000))]
        ),
        "tfidf_nb": Pipeline([("vec", TfidfVectorizer()), ("clf", MultinomialNB())]),
    }

    param_grids = {
        "tfidf_lr": {
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2, 5],
            "clf__C": [0.1, 1, 5],
        },
        "bow_lr": {
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2, 5],
            "clf__C": [0.1, 1, 5],
        },
        "tfidf_nb": {
            "vec__ngram_range": [(1, 1), (1, 2)],
            "vec__min_df": [1, 2, 5],
            "clf__alpha": [0.1, 0.5, 1.0],
        },
    }

    return pipelines, param_grids

# Using GridSearchCV to tune realhyperparameters
def tune_models(
    pipelines: dict[str, Pipeline],
    param_grids: dict[str, dict],
    X_train: pd.Series,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    print("\n===== Running GridSearchCV =====")

    best_models: dict[str, Pipeline] = {}
    for name, pipe in pipelines.items():
        print(f"\n>>> Tuning {name}")
        gs = GridSearchCV(
            pipe,
            param_grids[name],
            cv=5,
            scoring="f1",
            n_jobs=1,
        )
        gs.fit(X_train, y_train)
        print("Best params:", gs.best_params_)
        print("Best CV F1:", gs.best_score_)
        best_models[name] = gs.best_estimator_

    return best_models

# Using the F1-Score to pick and utilize the best model
def pick_best_model(best_models: dict[str, Pipeline], X_test: pd.Series, y_test: pd.Series) -> str:
    def f1_for(model: Pipeline) -> float:
        report = classification_report(y_test, model.predict(X_test), output_dict=True)
        return float(report["1"]["f1-score"])

    return max(best_models.keys(), key=lambda name: f1_for(best_models[name]))


def print_test_evaluation(best_models: dict[str, Pipeline], X_test: pd.Series, y_test: pd.Series) -> None:
    print("\n===== Final Test Set Evaluation =====")
    for name, model in best_models.items():
        preds = model.predict(X_test)
        print(f"\n==== {name.upper()} ====")
        print(classification_report(y_test, preds))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))


def show_confusion_matrix(best_name: str, model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Create error analysis for any users to understand misclassifications
def error_analysis(
    df: pd.DataFrame,
    best_model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    *,
    max_examples: int = 10,
) -> None:
    print("\n===== Error Analysis (some misclassified messages) =====")

    preds = pd.Series(best_model.predict(X_test), index=X_test.index)
    wrong_idx = preds.index[preds != y_test]
    if len(wrong_idx) == 0:
        print("No misclassified messages to display.")
        return

    sample_idx = pd.Series(wrong_idx).sample(min(max_examples, len(wrong_idx)), random_state=1)
    for i in sample_idx:
        print("TEXT:", df.loc[i, "text"])
        print("TRUE:", int(y_test.loc[i]), "PRED:", int(preds.loc[i]))
        print("-" * 80)


def predict_message(msg: str, *, model: Pipeline) -> str:
    msg_clean = normalize_text(msg, mode="lemma")
    pred = int(model.predict([msg_clean])[0])
    return "SPAM" if pred == 1 else "HAM"


def run_learning_curve_experiment(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    sizes: list[float] | None = None,
) -> pd.DataFrame:
    if sizes is None:
        sizes = [0.1, 0.3, 0.5, 1.0]

    pipeline = Pipeline(
        [
            ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )

    results: list[dict[str, float]] = []
    for size in sizes:
        n_samples = int(len(X_train) * size)

        if size < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=42)
            train_idx, _ = next(sss.split(X_train, y_train))
            X_sub = X_train.iloc[train_idx]
            y_sub = y_train.iloc[train_idx]
        else:
            X_sub, y_sub = X_train, y_train

        print(f"\nTraining on {size * 100:.0f}% of training data ({n_samples} samples)...")
        pipeline.fit(X_sub, y_sub)

        y_pred = pipeline.predict(X_test)
        f1 = float(f1_score(y_test, y_pred))
        print(f"F1-score on test set: {f1:.4f}")

        results.append(
            {
                "dataset_size_percent": size * 100,
                "n_samples": n_samples,
                "f1_score": f1,
            }
        )

    return pd.DataFrame(results)

# Plotting a learning curve to show the results visually
def plot_learning_curve(results: pd.DataFrame, *, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["n_samples"],
        results["f1_score"],
        marker="o",
        linewidth=2,
        markersize=8,
    )
    plt.xlabel("Training Set Size (Number of Samples)", fontsize=12)
    plt.ylabel("F1-Score (Test Set)", fontsize=12)
    plt.title("Learning Curve: Logistic Regression F1-Score vs Dataset Size", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.show()


def main() -> None:
    df = load_dataset(DATASET_PATH)
    df["clean_lemma"] = df["text"].apply(lambda t: normalize_text(t, mode="lemma"))

    X = df["clean_lemma"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipelines, param_grids = build_search_space()
    best_models = tune_models(pipelines, param_grids, X_train, y_train)

    print_test_evaluation(best_models, X_test, y_test)

    best_name = pick_best_model(best_models, X_test, y_test)
    best_model = best_models[best_name]
    print("\nBest model selected:", best_name)

    error_analysis(df, best_model, X_test, y_test)
    show_confusion_matrix(best_name, best_model, X_test, y_test)

    print("\n===== Learning Curve Experiment =====")
    learning_curve_results = run_learning_curve_experiment(X_train, y_train, X_test, y_test)

    results_csv_path = PROJECT_DIR / "learning_curve_results.csv"
    learning_curve_results.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")

    plot_path = PROJECT_DIR / "learning_curve_plot.png"
    plot_learning_curve(learning_curve_results, output_path=plot_path)

    print("\nLearning Curve Results:")
    print(learning_curve_results.to_string(index=False))

    print("\n===== Demo Predictions =====")
    print(predict_message("Congratulations! You have won a free iPhone. Click here!", model=best_model))
    print(predict_message("Hey, so are we meeting tomorrow at 5?", model=best_model))

# RUn the actual code to present its functionality
if __name__ == "__main__":
    main()
