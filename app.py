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
    pricedropdown = ''
    prediction = ''
    print(request.method)
    if request.method == "POST":

        countryList = [
            "Argentina",
            "Australia",
            "Austria",
            "Chile",
            "France",
            "Germany",
            "Italy",
            "Other",
            "Portugal",
            "Spain",
            "US"
        ]

        wineList = [
              "Bordeaux-style Red Blend",
              "Cabernet Sauvignon",
              "Chardonnay",
              "Malbec",
              "Merlet",
              "Other",
              "Pinot Noir",
              "Portuguese Red",
              "Red Blend",
              "Riesling",
              "Rose",
              "Sauvignon Blanc",
              "Syrah"       
        ]

        priceList = [

              "$10.00 and under",
              "$10.01-$12.00",
              "$12.01-$13.00",
              "$13.01-$14.00",
              "$14.01-$15.00",
              "$15.01-$16.00",
              "$16.01-$17.00",
              "$17.01-$18.00",
              "$18.01-$20.00",
              "$20.01-$22.00",
              "$22.01-$25.00",
              "$25.01-$30.00",
              "$30.01-$35.00",
              "$35.01-$40.00",
              "$40.01-$50.00",
              "$50.01 and over"
        ]


        
        winedropdown = request.form["winedropdown"]
        pricedropdown = request.form["pricedropdown"]
        ripe = 1 if "ripe" in request.form else 0
        crisp = 1 if "crisp" in request.form else 0
        bright =1 if "bright" in request.form else 0
        dry =1 if "dry" in request.form else 0
        full =1 if "full" in request.form else 0
        sweet =1 if "sweet" in request.form else 0
        fresh =1 if "fresh" in request.form else 0
        earthy =1 if "earthy" in request.form else 0
        bitter =1 if "bitter" in request.form else 0
        aftertaste =1 if "aftertaste" in request.form else 0
 

        prediction = []
        for x in wineList:
            if x == request.form["winedropdown"]:
                prediction.append(1)
            else:
                prediction.append(0)
        for x in priceList:
            if x == request.form["pricedropdown"]:
                prediction.append(1)
            else:
                prediction.append(0)        
        for x in countryList:
            if x == request.form["countrydropdown"]:
                prediction.append(1)
            else: 
                prediction.append(0)


        prefList = [ripe, crisp, bright, dry, full, sweet, fresh, earthy, bitter, aftertaste]
        prediction = prefList + prediction
        
        
        

        model = pickle.load(open("model.p", "rb"))
        p = model.predict([prediction])[0]
        
    else:
        print("Please choose selections!")
   
    return render_template("index.html",
                        P = p, 
                        Country = countrydropdown,
                        Variety = winedropdown,
                        Price = pricedropdown,
                        Ripe = ripe,
                        Crisp = crisp,
                        Bright = bright,
                        Dry = dry,
                        Full = full,
                        Sweet = sweet,
                        Fresh = fresh,
                        Earthy = earthy,
                        Bitter = bitter,
                        Aftertaste = aftertaste
                        )

if __name__ == "__main__":
    app.run(debug=True)





