import numpy as np
import pickle
from flask import Flask, render_template, request

model = pickle.load(open('model/KMeans_PCA.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

app = Flask(__name__, template_folder="templates")

@app.route('/')
def main():
  return render_template('main.html')

@app.route("/predict", methods=['POST'])
def predict():
  cash_advance = float(request.form["cash_advance"])
  purchases = float(request.form["purchases"])
       
  predict_list = [
    cash_advance, 
    purchases
  ]
  
  sample_scaled = scaler.fit_transform([predict_list])
  prediction = model.predict(sample_scaled)

  return render_template('main.html', prediction_text='Berada Pada Klaster: $ {}'.format(prediction))
        

if __name__ == '__main__':
    app.run(debug=True)

