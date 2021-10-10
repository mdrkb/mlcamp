import pickle

from flask import Flask
from flask import request
from flask import jsonify


dv_file = "dv.bin"
model_file = "model1.bin"


# Load the dict vectorizer
with open("dv.bin", "rb") as dv_bin:
    dv = pickle.load(dv_bin)
dv_bin.close()


# Load the model
with open("model2.bin", "rb") as model_bin:
    model = pickle.load(model_bin)
model_bin.close()


app = Flask("churn")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {"churn_probability": float(y_pred.round(3)), "churn": bool(churn)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
