from flask import Flask, render_template, request
import sklearn
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index(): 
    featureList = pd.read_csv("features.csv").to_dict("records")


    prediction = '' 
    if request.method == "POST":

        values = []
        for f in featureList: 
            name = f["Name"]
            f["Value"] = float(request.form[name])
            values.append(f["Value"])

        model = pickle.load(open("model.p", "rb"))

        print("values", values)

        prediction = model.predict([values])
        
        


    return render_template("index.html", featureList=featureList, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)