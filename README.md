# 💳 Loan Approval Prediction

Assessment-1 (ML) — a binary classification model that predicts whether a bank loan
application will be **approved** or **not approved**, based on the applicant's demographic and
financial details, including a **CIBIL Score**. Built with scikit-learn, deployed as a Streamlit
web app.

**Live app:** _add your Streamlit URL here after deploying_
**Base dataset:** [Loan Prediction Problem dataset](https://github.com/shrikant-temburwar/Loan-Prediction-Dataset) (Analytics Vidhya practice problem, 614 real rows)

## About the dataset

The original public dataset only has a **binary** `Credit_History` flag (1/0), not a real score.
This project simulates a realistic **CIBIL Score** (India's standard 300–900 credit bureau scale)
from that flag: applicants with `Credit_History = 1` get a score drawn from `Normal(760, 65)`,
those with `Credit_History = 0` get `Normal(560, 85)` — both clipped to `[300, 900]` and
deliberately overlapping (a "poor-history" applicant can still score decently, and vice versa),
rather than perfectly separated. `Credit_History` itself is then dropped from the dataset.

It's also expanded from 614 to **3,000 rows** via bootstrap resampling with jitter — the 614 real
rows are split into train/test *first*, then each side is independently expanded by resampling
its own real rows and jittering `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, and
`Cibil_Score` by ±15%. Splitting before expanding avoids a real row and its synthetic "sibling"
landing on opposite sides of the split, which would silently leak information into the test
metrics. Full method is documented in Sections 2a/2b of the notebook — **be ready to explain
this in your viva** if asked how the dataset/score was built; this is a documented simulation
and augmentation step, not 3,000 independently-collected real applications with real bureau
scores.

## Project structure

```
loan_approval/
├── data/
│   ├── train.csv                  # final 3,000-row dataset used for modeling (has Cibil_Score + a `split` column)
│   └── train_original_614.csv     # the original real dataset (still has the raw Credit_History flag)
├── Loan_Approval_Prediction.ipynb # full notebook: EDA, preprocessing, modeling, evaluation
├── model/
│   ├── loan_model.joblib          # saved sklearn Pipeline (preprocessing + tuned model)
│   └── feature_cols.joblib        # feature column list used at train time
├── app.py                         # Streamlit GUI (CIBIL Score input instead of Credit History)
└── requirements.txt
```

## Results (on the held-out 600-row test set)

| Model | CV F1 (train) | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.744 | – | – | – |
| Random Forest | 0.947 | – | – | – |
| Gradient Boosting | 0.881 | – | – | – |
| **Random Forest (tuned)** | **0.947** | **0.722** | **0.804** | **0.71** |

Random Forest was selected as the best baseline by cross-validated F1, then tuned with
`GridSearchCV`. **Note:** CV F1 looks higher than test F1 because, within the training set only,
a real row and its jittered bootstrap copies can land in different CV folds — the test set is
fully independent (see "About the dataset"), so the test-set numbers are the honest ones to
quote. `Cibil_Score` is the strongest single predictor (see the notebook's EDA and feature
importance plot) — as expected, it carries somewhat less "hard" separating power than the
original binary `Credit_History` flag did, since a realistic overlapping score is less
deterministic than a clean yes/no flag.

## Run locally

```bash
git clone <your-repo-url>
cd loan_approval
pip install -r requirements.txt
streamlit run app.py
```

## Re-train the model

Open and run `Loan_Approval_Prediction.ipynb` top to bottom — it regenerates `data/train.csv`
and re-saves `model/loan_model.joblib`, which `app.py` loads directly.

## Disclaimer

This is an academic demonstration. Predictions are not financial advice and should not be used
for real lending decisions.
