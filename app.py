from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)      # This must come before @app.route

model = joblib.load("model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    engine_temp_c = float(request.form["engine_temp_c"])
    coolant_temp_c = float(request.form["coolant_temp_c"])
    engine_load_percent = float(request.form["engine_load_percent"])
    vehicle_speed_kph = float(request.form["vehicle_speed_kph"])

    data = pd.DataFrame({
        "engine_temp_c": [engine_temp_c],
        "coolant_temp_c": [coolant_temp_c],
        "engine_load_percent": [engine_load_percent],
        "vehicle_speed_kph": [vehicle_speed_kph]
    })

    result = model.predict(data)

    return render_template("index.html", prediction=result[0])

if __name__ == "__main__":
    app.run(debug=True)