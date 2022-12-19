from flask import Flask,render_template,request
import pickle
import numpy as np

filename="C:\wamp64\www\phpless\Heart Attack ML Deployment (2)\Heart Attack ML Deployment\model.pkl"
model = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_heart_attack():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    thalach = int(request.form.get('thalach'))
    exang = int(request.form.get('exang'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))
    trtbps = int(request.form.get('trtbps'))
    oldpeak = int(float(request.form.get('oldpeak')))

    #prediction
    result = model.predict(np.array([age,sex,cp,thalach,exang,slope,ca,thal,trtbps,oldpeak]).reshape(1,10))

    if result[0] == 1:
        result = 'There is a chance of getting heart attack'
    else:
        result = 'No chance of getting a heart attack'

    return render_template('result.html',prediction=result)

if __name__ == '__main__':
    app.run(debug=True)