from flask import Flask, render_template, request
import sklearn
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index(): 

    if request.method == "POST":
        countrydropdown = request.form["countrydropdown"]
        winedropdown = request.form["winedropdown"]
        
 
        guess = [[countrydropdown, winedropdown]]

        model = pickle.load(open("model.p", "rb"))
        p = model.predict(guess)[0]
    
    else:
        print("Please choose selections.")
   
    return render_template("index.html",
                        P = p, 
                        countrydropdown = guess)

if __name__ == "__main__":
    app.run(debug=True)