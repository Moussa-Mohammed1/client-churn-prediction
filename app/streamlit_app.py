from pathlib import Path

import streamlit as st

from src.modeling import predict_customer


DEFAULT_BUNDLE = Path("data/processed/model_bundle.joblib")

st.set_page_config(page_title="Client churn", page_icon="📊", layout="wide")
st.title("Prédiction du churn client")
st.caption("Le cluster est calculé sans utiliser la variable cible Churn.")

if not DEFAULT_BUNDLE.exists():
    st.error(
        "Le modèle n'est pas disponible. Lancez d'abord "
        "`python -m src.train --input data/raw/telco-customer.csv`."
    )
    st.stop()

with st.form("customer_form"):
    left, right = st.columns(2)
    with left:
        senior = st.selectbox("Senior citizen", [0, 1])
        tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=100, value=12)
        monthly = st.number_input("Dépenses mensuelles", min_value=0.0, value=70.0)
        total = st.number_input("Dépenses totales", min_value=0.0, value=monthly * tenure)
        contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
        gender = st.selectbox("Genre", ["Female", "Male"])
        partner = st.selectbox("Partenaire", ["No", "Yes"])
        dependents = st.selectbox("Personnes à charge", ["No", "Yes"])
    with right:
        phone = st.selectbox("Téléphone", ["Yes", "No"])
        multiple = st.selectbox("Lignes multiples", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Sécurité en ligne", ["No internet service", "No", "Yes"])
        backup = st.selectbox("Sauvegarde en ligne", ["No internet service", "No", "Yes"])
        device = st.selectbox("Protection appareil", ["No internet service", "No", "Yes"])
        support = st.selectbox("Support technique", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox(
            "Streaming films", ["No internet service", "No", "Yes"]
        )
        paperless = st.selectbox("Facture électronique", ["No", "Yes"])
        payment = st.selectbox(
            "Moyen de paiement",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
    submitted = st.form_submit_button("Prédire")

if submitted:
    customer = {
        "customerID": "streamlit-customer",
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }
    try:
        result = predict_customer(DEFAULT_BUNDLE, customer)
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"Impossible de calculer la prédiction : {exc}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cluster", result["cluster"])
        c2.metric("Probabilité de churn", f"{result['churn_probability']:.1%}")
        c3.metric("Décision", "Risque élevé" if result["churn"] else "Risque faible")
