# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:23:29 2021

@author: SIMU
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("C:/Users/SIMU/Desktop/portfolio/salary prediction/model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("C:/Users/SIMU/Desktop/portfolio/salary prediction/templates/index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template(
        "C:/Users/SIMU/Desktop/portfolio/salary prediction/templates/index.html", prediction_text="Predicted Salary $ {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)