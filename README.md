# 💳 Loan Approval Prediction

Assessment-1 (ML) — a binary classification model that predicts whether a bank loan
application will be **approved** or **not approved**, based on the applicant's demographic and
financial details. Built with scikit-learn, deployed as a Streamlit web app.

**Live app:** _add your Streamlit URL here after deploying_
**Base dataset:** [Loan Prediction Problem dataset](https://github.com/shrikant-temburwar/Loan-Prediction-Dataset) (Analytics Vidhya practice problem, 614 real rows)

## About the dataset

The original public dataset has 614 rows. It's expanded to **3,000 rows** for this assignment
using **bootstrap resampling with jitter** — the 614 real rows are split into train/test *first*,
then each side is independently expanded by resampling its own real rows and adding a small
random ±15% jitter to `ApplicantIncome`, `CoapplicantIncome`, and `LoanAmount`. Splitting before
expanding avoids a real row and its synthetic "sibling" landing on opposite sides of the split
(which would silently leak information and inflate test metrics). Full method and reasoning is
documented in Section 2b of the notebook — **be ready to explain this in your viva** if asked how
the dataset was built; it is a documented augmentation step, not 3,000 independently-collected
real applications.

## Project structure

```
loan_approval/
├── data/
│   ├── train.csv                  # final 3,000-row dataset used for modeling (has a `split` column)
├── Loan_Approval_Prediction.ipynb # full notebook: EDA, preprocessing, modeling, evaluation
├── model/
│   ├── loan_model.joblib          # saved sklearn Pipeline (preprocessing + tuned model)
│   └── feature_cols.joblib        # feature column list used at train time
├── app.py                         # Streamlit GUI
└── requirements.txt
```

## Results (on the held-out 600-row test set)

| Model | CV F1 (train) | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.829 | – | – | – |
| Random Forest | 0.971 | – | – | – |
| Gradient Boosting | 0.906 | – | – | – |
| **Random Forest (tuned)** | **0.971** | **0.793** | **0.856** | **0.75** |

Random Forest was selected as the best baseline by cross-validated F1, then tuned with
`GridSearchCV`. **Note:** the CV F1 above looks much higher than the test F1 because, within the
training set only, a real row and its jittered bootstrap copies can land in different CV folds —
the test set itself is fully independent (see "About the dataset"), so the test-set numbers are
the honest ones to quote. Class imbalance is handled with `class_weight="balanced"`; skewed
income/loan columns are log-transformed. Full plots and the classification report are in the
notebook.

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

