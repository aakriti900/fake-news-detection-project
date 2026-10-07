"""Train and evaluate fake news detection models. Run: python train.py"""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

os.makedirs("outputs", exist_ok=True)


def load_data():
    """Use Kaggle Fake.csv / True.csv if present, else data/news.csv."""
    if os.path.exists("data/Fake.csv") and os.path.exists("data/True.csv"):
        fake = pd.read_csv("data/Fake.csv"); fake["label"] = "FAKE"
        real = pd.read_csv("data/True.csv"); real["label"] = "REAL"
        df = pd.concat([fake, real])
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        print("Loaded Kaggle dataset")
        return df[["text", "label"]].sample(frac=1, random_state=1).reset_index(drop=True)
    if not os.path.exists("data/news.csv"):
        import make_sample_data  # noqa: F401  (creates data/news.csv)
    print("Loaded demo dataset data/news.csv")
    return pd.read_csv("data/news.csv")


df = load_data().dropna()
print(f"\nDataset size: {len(df)}")
print(df["label"].value_counts().to_string(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Passive Aggressive (SGD)": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, random_state=42),
}

results, best_name, best_acc, best_pipe = {}, None, 0, None
for name, clf in models.items():
    pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7,
                                               ngram_range=(1, 2))),
                     ("clf", clf)])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    results[name] = acc
    print(f"{name:22s} accuracy: {acc:.4f}")
    if acc > best_acc:
        best_name, best_acc, best_pipe = name, acc, pipe

print(f"\nBest model: {best_name} ({best_acc:.2%})\n")
pred = best_pipe.predict(X_test)
print(classification_report(y_test, pred))

# Confusion matrix
cm = confusion_matrix(y_test, pred, labels=["FAKE", "REAL"])
ConfusionMatrixDisplay(cm, display_labels=["FAKE", "REAL"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("outputs/confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# Model comparison
plt.figure(figsize=(7, 4))
plt.bar(results.keys(), results.values(), color=["#4c78a8", "#f58518", "#54a24b"])
plt.ylim(0, 1.05); plt.ylabel("Accuracy"); plt.title("Model Comparison")
plt.xticks(rotation=10)
plt.savefig("outputs/model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Top words (only for linear models)
if hasattr(best_pipe.named_steps["clf"], "coef_"):
    names = best_pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = best_pipe.named_steps["clf"].coef_[0]
    classes = best_pipe.named_steps["clf"].classes_
    order = coefs.argsort()
    print(f"Top words indicating {classes[0]}:", [names[i] for i in order[:10]])
    print(f"Top words indicating {classes[1]}:", [names[i] for i in order[-10:]])

joblib.dump(best_pipe, "outputs/model.joblib")
print("\nSaved: outputs/model.joblib, confusion_matrix.png, model_comparison.png")
print("Now run: python predict.py")
