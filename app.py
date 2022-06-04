from flask import Flask, render_template, request
import sklearn
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index(): 
    p = ''
    countrydropdown = ''
    winedropdown = ''
    preference = ''
    if request.method == "POST":
        countrydropdown = request.form["countrydropdown"]
        winedropdown = request.form["winedropdown"]
        preference =request.form.getlist('preference')
 
        prediction = [[countrydropdown, winedropdown, preference]]
        
        model = pickle.load(open("model.p", "rb"))
        p = model.predict(prediction)[0]
    
    else:
        print("Please choose selections!")
   
    return render_template("index.html",
                        P = p, 
                        Country = countrydropdown,
                        Variety = winedropdown,
                        Preference = preference
                        )

if __name__ == "__main__":
    app.run(debug=True)