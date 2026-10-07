"""Interactive predictor. Run: python predict.py"""
import os
import sys
import joblib

if not os.path.exists("outputs/model.joblib"):
    sys.exit("Model not found. Run 'python train.py' first.")

model = joblib.load("outputs/model.joblib")

samples = [
    "Central bank holds rates steady as inflation data shows a slight decline",
    "SHOCKING: Big Pharma secretly hiding the truth about vaccination, insiders reveal!!!",
]
print("=== Sample predictions ===")
for s in samples:
    print(f"[{model.predict([s])[0]}] {s}")

print("\n=== Try your own (type 'q' to quit) ===")
while True:
    text = input("\nEnter news headline/text: ").strip()
    if text.lower() in {"q", "quit", "exit"}:
        break
    if not text:
        continue
    label = model.predict([text])[0]
    msg = f"Prediction: {label}"
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([text])[0].max()
        msg += f"  (confidence {prob:.1%})"
    print(msg)
