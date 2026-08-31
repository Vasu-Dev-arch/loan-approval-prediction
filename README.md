# 💳 Loan Approval Prediction

Assignment-1 (ML) — a binary classification model that predicts whether a bank loan
application will be **approved** or **not approved**, based on the applicant's demographic and
financial details, including a **CIBIL Score**. Built with scikit-learn, deployed as a Streamlit
web app.

**Live app:** https://loan-approval-prediction-123.streamlit.app/


## Project structure

```
loan_approval/
├── data/
│   ├── train.csv                  # final 3,000-row dataset used for modeling (has Cibil_Score + a `split` column)
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

