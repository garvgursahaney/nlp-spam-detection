import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse

from pathlib import Path
import io
import zipfile
import urllib.request

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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


# Thisis where the dataset was loaded
ensure_dataset(DATASET_PATH)
df = pd.read_csv(DATASET_PATH)
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

stop_words = set(ENGLISH_STOP_WORDS)
stemmer = PorterStemmer() if PorterStemmer else None
lemmatizer = WordNetLemmatizer() if WordNetLemmatizer else None


def clean_text(text: str, mode="stem"):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]

    if mode == "stem" and stemmer:
        tokens = [stemmer.stem(w) for w in tokens]
    elif mode == "lemma" and lemmatizer:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)


# I attempted to prepare clean texts as variables for ease of use and understanding
df["clean_stem"] = df["text"].apply(lambda t: clean_text(t, "stem"))
df["clean_lemma"] = df["text"].apply(lambda t: clean_text(t, "lemma"))

X = df["clean_lemma"]  
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Training using Model Pipelines & Parameter Grids
pipelines = {
    "tfidf_lr": Pipeline([
        ("vec", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "bow_lr": Pipeline([
        ("vec", CountVectorizer()),
        ("clf", LogisticRegression(max_iter=2000))
    ]),
    "tfidf_nb": Pipeline([
        ("vec", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])
}

param_grids = {
    "tfidf_lr": {
        "vec__ngram_range": [(1,1), (1,2)],
        "vec__min_df": [1, 2, 5],
        "clf__C": [0.1, 1, 5]
    },
    "bow_lr": {
        "vec__ngram_range": [(1,1), (1,2)],
        "vec__min_df": [1, 2, 5],
        "clf__C": [0.1, 1, 5]
    },
    "tfidf_nb": {
        "vec__ngram_range": [(1,1), (1,2)],
        "vec__min_df": [1, 2, 5],
        "clf__alpha": [0.1, 0.5, 1.0]
    }
}

# I used GridSearchCV for training as well and returned a success
print("\n===== Running GridSearchCV =====")

best_models = {}

for name in pipelines:
    print(f"\n>>> Tuning {name}")
    gs = GridSearchCV(
        pipelines[name],
        param_grids[name],
        cv=5,
        scoring="f1",   
        n_jobs=1        
    )
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV F1:", gs.best_score_)

    best_models[name] = gs.best_estimator_

# After that came the final test set evaluation
print("\n===== Final Test Set Evaluation =====")

for name, model in best_models.items():
    preds = model.predict(X_test)
    print(f"\n==== {name.upper()} ====")
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

# Trial to select the best model based on F1-score
best_name = max(
    best_models.keys(),
    key=lambda k: classification_report(
        y_test, best_models[k].predict(X_test), output_dict=True
    )["1"]["f1-score"]
)

best_model = best_models[best_name]


print("\nBest model selected:", best_name)

# It was also important for me to include an error analysis system
print("\n===== Error Analysis (some misclassified messages) =====")

preds = best_model.predict(X_test)
errors = X_test[preds != y_test]

if len(errors) == 0:
    print("No misclassified messages to display.")
else:
    for i, txt in errors.sample(min(10, len(errors)), random_state=1).items():
        print("TEXT:", df.loc[i, "text"])
        print("TRUE:", y_test.loc[i], "PRED:", preds[list(X_test.index).index(i)])
        print("-" * 80)

#  Confusion Matrix Plot
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.title(f"Confusion Matrix ({best_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Predicition Function of any new messages that may arrive
def predict_message(msg):
    msg = clean_text(msg, "lemma")
    pred = best_model.predict([msg])[0]
    return "SPAM" if pred == 1 else "HAM"


print("\n===== Learning Curve Experiment =====")

def run_learning_curve_experiment(X_train, y_train, X_test, y_test, sizes=[0.1, 0.3, 0.5, 1.0]):
    """
    Train Logistic Regression on stratified subsets of training data
    and evaluate F1-score on test set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        sizes: List of fractions of training data to use (default: [0.1, 0.3, 0.5, 1.0])
    
    Returns:
        DataFrame with results (dataset_size, n_samples, f1_score)
    """
    results = []
    
    # Towards the end, I could properly create a pipeline with the use of Logistic Regression and TF-IDF
    pipeline = Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0))
    ])
    
    for size in sizes:
        # Calculate actual number of samples
        n_samples = int(len(X_train) * size)
        
        # Use stratified sampling to get subset
        if size < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=42)
            train_idx, _ = next(sss.split(X_train, y_train))
            X_train_subset = X_train.iloc[train_idx]
            y_train_subset = y_train.iloc[train_idx]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        print(f"\nTraining on {size*100:.0f}% of training data ({n_samples} samples)...")
        
        # Train model
        pipeline.fit(X_train_subset, y_train_subset)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        
        print(f"F1-score on test set: {f1:.4f}")
        
        results.append({
            'dataset_size_percent': size * 100,
            'n_samples': n_samples,
            'f1_score': f1
        })
    
    return pd.DataFrame(results)

# Run learning curve experiment
learning_curve_results = run_learning_curve_experiment(X_train, y_train, X_test, y_test)

# All the results are programmed to save as a CSV file 
results_csv_path = PROJECT_DIR / "learning_curve_results.csv"
learning_curve_results.to_csv(results_csv_path, index=False)
print(f"\nResults saved to: {results_csv_path}")

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(learning_curve_results['n_samples'], learning_curve_results['f1_score'], 
         marker='o', linewidth=2, markersize=8)
plt.xlabel('Training Set Size (Number of Samples)', fontsize=12)
plt.ylabel('F1-Score (Test Set)', fontsize=12)
plt.title('Learning Curve: Logistic Regression F1-Score vs Dataset Size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()


plot_path = PROJECT_DIR / "learning_curve_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# Display results table
print("\nLearning Curve Results:")
print(learning_curve_results.to_string(index=False))

plt.show()

print("\n===== Demo Predictions =====")
print(predict_message("Congratulations! You have won a free iPhone. Click here!"))
print(predict_message("Hey, so are we meeting tomorrow at 5?"))
