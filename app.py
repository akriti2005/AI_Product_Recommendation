from flask import Flask, render_template, request
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# SHAP explainer
explainer = shap.Explainer(model)

# Load dataset
data = pd.read_csv("ecommerce_data.csv")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    user_data = {
        "Number of clicks on similar products": float(request.form['clicks']),
        "Number of similar products purchased so far": float(request.form['purchased']),
        "Average rating given to similar products": float(request.form['avg_rating']),
        "Median purchasing price (in rupees)": float(request.form['median_price']),
        "Rating of the product": float(request.form['rating']),
        "Customer review sentiment score (overall)": float(request.form['sentiment']),
        "Price of the product": float(request.form['price'])
    }

    # Convert input to dataframe
    df = pd.DataFrame([user_data])

    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)

    # Prediction
    prediction = model.predict(df)[0]

    # -----------------------
    # SHAP Explanation
    # -----------------------

    shap_values = explainer(df)

    shap_dict = dict(zip(df.columns, shap_values.values[0]))

    top_features = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    explanation = [
        f"{feature} impact: {round(value,3)}"
        for feature, value in top_features
    ]

    # -----------------------
    # AI explanation sentence
    # -----------------------

    ai_reason = (
        "This product is recommended because the user interacts frequently "
        "with similar products and the product rating and sentiment score are high."
    )

    # -----------------------
    # SHAP Graph
    # -----------------------

    plt.figure()

    shap.plots.waterfall(shap_values[0], show=False)

    plot_path = "static/shap_plot.png"

    plt.savefig(plot_path, bbox_inches="tight")

    plt.close()

    # -----------------------
    # Recommended products
    # -----------------------

    top_products = data.sort_values(
        "Probability for the product to be recommended to the person",
        ascending=False
    ).head(5)

    products = top_products["Brand of the product"].tolist()

    return render_template(
        "index.html",
        prediction=round(prediction, 2),
        products=products,
        explanation=explanation,
        shap_plot=plot_path,
        ai_reason=ai_reason
    )


if __name__ == "__main__":
    
