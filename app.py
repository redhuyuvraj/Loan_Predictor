from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("loan_approval_model (1).pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract values from form
        age = int(request.form["age"])
        experience = int(request.form["experience"])
        income = int(request.form["income"])
        family = int(request.form["family"])
        ccavg = float(request.form["ccavg"])
        education = int(request.form["education"])
        mortgage = int(request.form["mortgage"])
        securities_account = int(request.form["securities_account"])
        cd_account = int(request.form["cd_account"])
        online = int(request.form["online"])
        creditcard = int(request.form["creditcard"])

        # Prepare feature vector
        features = np.array([[age, experience, income, family, ccavg,
                              education, mortgage, securities_account,
                              cd_account, online, creditcard]])

        # Predict
        prediction = model.predict(features)[0]
        result = "Loan Approved" if prediction == 1 else "Loan Not Approved"

        return render_template("result.html", prediction=result)
    
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
