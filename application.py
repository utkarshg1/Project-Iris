from flask import Flask, render_template, request, jsonify
import pandas as pd
from custom_def import load_dataframe, load_pickle

# Intilalize flask instance
application = Flask(__name__)
app = application

# View The homepage
@app.route('/')
def home_page():
    return render_template('index.html')

# Predciton Route
@app.route('/predict',methods=['POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        sep_len = float(request.form.get('sepal_length'))
        sep_wid = float(request.form.get('sepal_width'))
        pet_len = float(request.form.get('petal_length'))
        pet_wid = float(request.form.get('petal_width'))

        pre_path = 'notebooks/Preprocessor.pkl'
        xnew = load_dataframe(sep_len,sep_wid,pet_len,pet_wid,pre_path)

        le_path = 'notebooks/LabelEnc.pkl'
        model_path = 'notebooks/model.pkl'

        le = load_pickle(le_path)
        model = load_pickle(model_path)

        pred = model.predict(xnew)
        pred_lb = le.inverse_transform(pred)[0]

        prob = model.predict_proba(xnew).max()

        prediction = f'{pred_lb} with Probability : {prob:.4f}'

        return render_template('index.html',prediction=prediction)
    
# Creating an API
@app.route('/predict_api',methods=['POST'])
def predict_point():
    if request.method=='POST':
        sep_len = request.json['sepal_length']
        sep_wid = request.json['sepal_width']
        pet_len = request.json['petal_length']
        pet_wid = request.json['petal_width']

        pre_path = 'notebooks/Preprocessor.pkl'
        xnew = load_dataframe(sep_len, sep_wid, pet_len, pet_wid, pre_path)

        le_path = 'notebooks/LabelEnc.pkl'
        model_path = 'notebooks/model.pkl'

        le = load_pickle(le_path)
        model = load_pickle(model_path)

        pred = model.predict(xnew)
        pred_lb = le.inverse_transform(pred)[0]

        prob = model.predict_proba(xnew).max()

        return jsonify({'Prediction':pred_lb,
                        'Probability':prob})


if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)