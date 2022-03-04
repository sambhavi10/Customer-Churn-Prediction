from flask import Flask, render_template, request
# import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

#create a Flask object
app  = Flask("customer_segmentation")

model = pickle.load(open('cust_segmentation.pkl','rb'))

#define the route(basically url) to which we need http request
#HTTP GET request method
@app.route('/',methods = ['GET'])
#create a function that will direct to index.html
def Home():
    return render_template('index.html')

#creating object for StandardScaler(used for scaling data)
standard_to = StandardScaler()

#HTTP POST request method
#defining the route for this post method
@app.route("/predict" ,methods = ['POST'])

def predict():
    if request.method == 'POST':

        Balance = float(request.form['Balance'])
        Purchases = float(request.form['Purchases'])
        Oneoff_Purchases = float(request.form['Oneoff_Purchases'])
        Installment_Purchases = float(request.form['Installment_Purchases'])
        Cash_Advance = float(request.form['Cash_Advance'])
        Purchases_Frequency = float(request.form['Purchases_Frequency'])
        Oneoff_Purchases_Frequency = float(request.form['Oneoff_Purchases_Frequency'])
        Purchases_Installments_Frequency = float(request.form['Purchases_Installments_Frequency'])
        Cash_Advance_Frequency = float(request.form['Cash_Advance_Frequency'])
        Cash_Advance_Trx = float(request.form['Cash_Advance_Trx'])
        Purchases_Trx = float(request.form['Purchases_Trx'])
        Credit_Limit =float( request.form['Credit_Limit'])
        Payments = float(request.form['Payments'])
        
        prediction = model.predict([[Balance, Purchases, Oneoff_Purchases, Installment_Purchases, Cash_Advance, Purchases_Frequency, Oneoff_Purchases_Frequency,Purchases_Installments_Frequency,Cash_Advance_Frequency,Cash_Advance_Trx,Purchases_Trx,Credit_Limit,Payments]])

        return render_template('index.html', prediction_text = "The predicted cluster is {} ".format(prediction))


    #html form to be displayed on screen when no values are inserted; without any output or prediction
    else:
        return render_template('index.html')

if __name__== "__main__":
    app.run(debug = True, port = 8000)

    
