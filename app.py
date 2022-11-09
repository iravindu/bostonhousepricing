import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# define flask app
app = Flask(__name__)

# load pickle file (model)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# by default when you open the flask app it will refirect to this home page
@app.route('/')
def home():
    return render_template('home.html') 

# creating the predict api
# this is post request bcz we give input. this will capture the input
# then we wiil go and give that input to our model.
# then our model will give the output
@app.route('/predict_api', methods = ['POST']) 
def predict_api():
    # whenever I hit predict_api making sure that the input is in json format
    data = request.json['data'] 
    print(data)

    # first of all we should do standardization
    print(np.array(list(data.values())).reshape(1,-1))

    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))

    output = regmodel.predict(new_data)
    # output will be in a 2d array. so we are taking the first value
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The house price prediction is {}".format(output))
    
if __name__ == "__main__":
    app.run(debug = True)