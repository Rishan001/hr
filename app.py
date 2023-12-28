from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def index():

    return render_template("detail.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
     
    form_values = request.form.values()
    int_features = [float(x) for x in form_values if x.replace('.', '', 1).isdigit()]

    final=[np.array(int_features)]

    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)


    if output >= str(0.85):
            return render_template('pred.html', pred='High Chance for Promotion')
    else:
            return render_template('notpred.html', pred='Low Chance for Promotion')



# Your other routes and configurations...

if __name__ == '__main__':
    app.run(debug=True)