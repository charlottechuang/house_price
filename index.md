import os
from tensorflow import keras
from flask import Flask,request,render_template,redirect,url_for
import joblib
from werkzeug.datastructures import ImmutableMultiDict
import pandas as pd

app=Flask(__name__)

model=joblib.load('clf.pkl')

def get_form_to_result(form):
    form=pd.DataFrame(form,index=[0]) #
    prediction=model.predict(form)
    return prediction

@app.route("/")
def formPage():
    return render_template('House_Feature_Form.html')


@app.route('/submit',methods=['POST'])
def submit():
    if request.method=='POST':
        global form_dct
        form_dct=request.values.to_dict()   #取得使用者上傳的表單並轉換成字典形式
        return redirect(url_for('success'))

@app.route('/success')
def success():
    prediction=get_form_to_result(form_dct)
    return render_template('result.html',prediction = prediction)

if __name__ == "__main__":
    app.run()
