from flask import Flask, request, render_template
import pickle
import numpy as np
model=pickle.load(open('model2.pkl', 'rb'))
app=Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=['post'])
def predict():
    dis =int( request.form.get('dis'))
    hor =int( request.form.get('hor'))
    weight =int( request.form.get('weight'))
    acce = int(request.form.get('acce'))
    input = [dis,hor,weight,acce]
    output = model.predict([input])
    output = np.round(output[0], 2).item()
    output = str(output)+"mpg"
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)