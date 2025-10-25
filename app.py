import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Load saved models
# ----------------------------
kmeans = joblib.load('/Users/prashanth45/Downloads/kmeans_model.pkl')
scaler = joblib.load('/Users/prashanth45/Downloads/scaler.pkl')
xgb_model = joblib.load('/Users/prashanth45/Downloads/xgb_model.pkl')

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ecommerce Purchase Behaviour", layout="centered")
st.title('🛍️ Ecommerce Purchase Behaviour')
st.write('Enter customer details to predict **customer segment** and **high-value potential**.')

# ----------------------------
# Input fields
# ----------------------------
Age = st.number_input("Age of Customer:", min_value=18, max_value=100, value=40)
Income_Level = st.selectbox("Income Level of Customer", ["Low", "Medium", "High"])
Purchase_Amount = st.number_input("Purchase Amount:", min_value=0, max_value=10_000_000, value=100_000)
Frequency_of_Purchase = st.number_input("Frequency of Purchase:", min_value=0, max_value=1000, value=10)
Overall_Satisfaction = st.number_input("Overall Satisfaction (0-10):", min_value=0, max_value=10, value=4)

# Encode categorical input (consistent with LabelEncoder used during training)
income_mapping = {"Low": 0, "Medium": 1, "High": 2}
Income_Level_encoded = income_mapping[Income_Level]

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame({
    'Age': [Age],
    'Income_Level': [Income_Level_encoded],
    'Purchase_Amount': [Purchase_Amount],
    'High_Value_Customer': [0],  # placeholder if scaler expects it
    'Frequency_of_Purchase': [Frequency_of_Purchase],
    'Overall_Satisfaction': [Overall_Satisfaction]
})

# Align columns with what the scaler expects
try:
    input_data = input_data[scaler.feature_names_in_]
except AttributeError:
    st.warning("⚠️ The loaded scaler doesn’t have 'feature_names_in_'. Make sure you trained it with a DataFrame.")

# Scale the input
input_scaled = scaler.transform(input_data)

# ----------------------------
# Predict Button
# ----------------------------
if st.button('🔍 Predict Segment & Value'):
    # --- KMeans Prediction ---
    cluster = kmeans.predict(input_scaled)[0]

    # --- XGBoost Prediction ---
    xgb_pred = xgb_model.predict(input_scaled)[0]
    xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]

    # ----------------------------
    # Display Results
    # ----------------------------
    st.success(f"Predicted Segment: {cluster}")
    st.markdown(f"**High-Value Customer Prediction:** {'✅ Yes' if xgb_pred == 1 else '❌ No'}")
    st.progress(int(xgb_prob * 100))

    # ----------------------------
    # Interpretation Section
    # ----------------------------
    st.markdown("---")
    st.subheader("📊 Interpretation")

    # Cluster insights (customize as per your cluster_summary)
    if cluster == 0:
        st.info("🧩 Segment 0: Young customers with low income but high spending tendencies.")
    elif cluster == 1:
        st.info("🧩 Segment 1: High-income, frequent buyers — loyal and premium customers.")
    elif cluster == 2:
        st.info("🧩 Segment 2: Moderate income, average purchase frequency — potential to grow.")
    elif cluster == 3:
        st.info("🧩 Segment 3: High spending but low satisfaction — at risk of churn.")
    else:
        st.warning("Segment not clearly defined in training data.")

    st.write(f"🧠 Probability of being a High-Value Customer: **{xgb_prob:.2%}**")
