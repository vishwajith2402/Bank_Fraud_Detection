import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# page setup
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# gradient dark theme
st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#020617,#0f172a);
color:white;
}

</style>
""", unsafe_allow_html=True)

# login system
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.title("Secure Login")

    username = st.text_input("Username")

    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "1234":

            st.session_state.login = True
            st.rerun()

        else:

            st.error("Invalid login")

    st.stop()

# load model
model = joblib.load("model.pkl")

st.title("AI Bank Fraud Detection Dashboard")

st.write("Machine Learning powered fraud detection")

# layout
col1,col2 = st.columns(2)

with col1:

    st.subheader("Transaction Info")

    time = st.number_input("Time", value=10000.0)

    amount = st.number_input("Amount", value=1000.0)

with col2:

    st.subheader("Behaviour Pattern")

    v1 = st.number_input("V1", value=0.0)

    v2 = st.number_input("V2", value=0.0)

    v3 = st.number_input("V3", value=0.0)

    v4 = st.number_input("V4", value=0.0)

    v5 = st.number_input("V5", value=0.0)

remaining_features = [0]*23

features = [
    time,
    v1,v2,v3,v4,v5,
    *remaining_features,
    amount
]

# history
if "history" not in st.session_state:

    st.session_state.history = []

# prediction
if st.button("Analyze Transaction"):

    X = np.array(features).reshape(1,-1)

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    risk = probability * 100

    st.subheader("Prediction Result")

    col3,col4,col5 = st.columns(3)

    col3.metric("Fraud Risk %", f"{risk:.2f}%")

    col4.metric("Safety %", f"{100-risk:.2f}%")

    col5.metric(
        "Risk Level",
        "High" if risk>70 else "Medium" if risk>30 else "Low"
    )

    # gauge meter
    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=risk,

        title={'text': "Fraud Risk"},

        gauge={

            'axis':{'range':[0,100]},

            'bar':{'color':"red"},

            'steps':[

                {'range':[0,30],'color':"green"},

                {'range':[30,70],'color':"orange"},

                {'range':[70,100],'color':"red"}

            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

    if risk>70:

        st.error("High Risk Fraud Transaction")

    elif risk>30:

        st.warning("Medium Risk Transaction")

    else:

        st.success("Normal Transaction")

    st.session_state.history.append({

        "Amount":amount,

        "Risk %":risk,

        "Result":"Fraud" if prediction==1 else "Safe"
    })

st.divider()

# charts
col6,col7 = st.columns(2)

with col6:

    st.subheader("Confusion Matrix")

    st.image("confusion_matrix.png")

with col7:

    st.subheader("Feature Importance")

    st.image("feature_importance.png")

st.divider()

# history table
st.subheader("Transaction History")

if st.session_state.history:

    hist = pd.DataFrame(st.session_state.history)

    st.dataframe(hist)

    st.bar_chart(hist["Risk %"])

st.divider()

# batch prediction
st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV")

if file:

    df = pd.read_csv(file)

    preds = model.predict(df)

    df["Prediction"] = preds

    st.write(df.head())

    csv = df.to_csv(index=False)

    st.download_button(

        "Download Results",

        csv,

        "fraud_predictions.csv"

    )