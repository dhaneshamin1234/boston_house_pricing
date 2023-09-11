import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


# initialize flask
app = Flask(__name__)

# load model
rfr_model = pickle.load(open('boston_rfr_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# defining routes


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = rfr_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict2', methods=['POST'])
def predict2():
    print(request.form.values())
    data = [float(x) for x in request.form.values()]
    print(data)
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    # print(final_input)
    output = rfr_model.predict(final_input)[0]
    output = round(output)
    print(output)
    # print('test', rfr_model.predict(final_input))
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
