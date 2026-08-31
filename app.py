import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💳", layout="centered")

MODEL_PATH = "model/loan_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("💳 Loan Approval Prediction")
st.caption("Educational ML demonstration using a tuned Random Forest classifier.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=5)
        loan_term = st.number_input("Loan Amount Term (days)", min_value=12, value=360, step=12)
        credit_history = st.selectbox("Credit History", ["Good (has repaid past debts)", "Poor / None"])

    submitted = st.form_submit_button("Predict")

if submitted:
    credit_history_val = 1 if credit_history.startswith("Good") else 0

    row = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome_log": np.log1p(applicant_income),
        "CoapplicantIncome_log": np.log1p(coapplicant_income),
        "LoanAmount_log": np.log1p(loan_amount),
        "Loan_Amount_Term": loan_term,
        "Credit_History": float(credit_history_val),
        "Property_Area": property_area,
    }])

    proba_approved = model.predict_proba(row)[0][1]
    prediction = model.predict(row)[0]

    st.subheader("Estimated loan approval probability")
    st.markdown(f"## {proba_approved * 100:.2f}%")

    if prediction == 1:
        st.success("Model prediction: **Loan Approved**")
    else:
        st.error("Model prediction: **Loan Not Approved**")

    st.warning("This prediction is for an academic demonstration and is not financial advice.")

    with st.expander("See what you entered"):
        st.dataframe(row)
