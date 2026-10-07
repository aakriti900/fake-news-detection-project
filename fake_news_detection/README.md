# Fake News Detection (TF-IDF + Machine Learning)

## Run in VS Code
1. Open this folder in VS Code (File > Open Folder).
2. Open the terminal (Ctrl + `) and install packages:
   `pip install -r requirements.txt`
3. Train and see results: `python train.py`
4. Try your own headlines: `python predict.py`

Charts are saved in the `outputs/` folder (open them in VS Code).

## Use a real dataset (recommended)
Download "Fake and Real News Dataset" from Kaggle (ISOT: Fake.csv and True.csv),
place both files in `data/`, and run `python train.py` again. It is auto-detected.

## How it works
Text -> TF-IDF features (unigrams + bigrams) -> classifier (Logistic Regression,
Naive Bayes, Passive Aggressive) -> best model is saved and used for predictions.

Note: the built-in demo dataset is synthetic, so accuracy will be near 100%.
Real data gives realistic scores (about 95-99%).
