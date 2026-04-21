import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# Load model + data
# -----------------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
data = pd.read_csv("ecommerce_data.csv")

# SHAP explainer (safe mode)
explainer = shap.Explainer(model)

# -----------------------------
# UI
# -----------------------------
st.title("AI Product Recommendation System")

st.write("Enter customer/product details below:")

clicks = st.number_input("Clicks on similar products", min_value=0.0)
purchased = st.number_input("Similar products purchased", min_value=0.0)
avg_rating = st.number_input("Average rating given to similar products", min_value=0.0)
median_price = st.number_input("Median purchasing price (in rupees)", min_value=0.0)
rating = st.number_input("Product rating", min_value=0.0)
sentiment = st.number_input("Customer sentiment score", min_value=0.0)
price = st.number_input("Product price", min_value=0.0)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Recommend Product"):

    # Create input dataframe
    user_data = {
        "Number of clicks on similar products": clicks,
        "Number of similar products purchased so far": purchased,
        "Average rating given to similar products": avg_rating,
        "Median purchasing price (in rupees)": median_price,
        "Rating of the product": rating,
        "Customer review sentiment score (overall)": sentiment,
        "Price of the product": price
    }

    df = pd.DataFrame([user_data])

    # Match training features
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    # Prediction
    prediction = model.predict(df)[0]

    st.success(f"Prediction Score: {round(prediction, 2)}")

    # -----------------------------
    # Top Products Recommendation
    # -----------------------------
    top_products = data.sort_values(
        "Probability for the product to be recommended to the person",
        ascending=False
    ).head(5)

    st.subheader("Top Recommended Products")
    st.write(top_products["Brand of the product"].tolist())

    # -----------------------------
    # SHAP Explanation (safe display)
    # -----------------------------
    try:
        shap_values = explainer(df)

        st.subheader("Feature Importance (SHAP)")

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation could not be displayed in cloud environment.")
