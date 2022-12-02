import numpy as np
import pickle
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open('model/RandomForest_Tuning.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def main():
  return render_template('main.html')

if __name__ == '__main__':
  app.run(debug=True)

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Age = request.form['Age']
        Gender = request.form['Gender']
        Creatine = request.form['Creatine']
        Eject = request.form['Eject']
        Platelets = request.form['Platelets']
        Serum_Creatinine = request.form['Serum_Creatinine']
        Serum_Sodium = request.form['Serum_Sodium']
        Anaemia = request.form['Anaemia']
        Diabetes = request.form['Diabetes']
        High_Blood = request.form['High_Blood']
        Smoking = request.form['Smoking']

        predict_list = [Age, Anaemia, Creatine, Diabetes, Eject, High_Blood, Platelets, Serum_Creatinine, Serum_Sodium, Gender, Smoking]
        sample = np.array(predict_list, dtype=float).reshape(1,-1)
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        
        output = {0: 'Tidak Meninggal.', 1: 'Meninggal'}
        
        if prediction == 0:
            return render_template('main.html', prediction_text='Pasien {}'.format(output[prediction[0]]))
        elif prediction == 1:
            return render_template('main2.html', prediction_text='Pasien {}'.format(output[prediction[0]]))
        else:
            return render_template('main.html')
    else:
        return render_template('main.html')