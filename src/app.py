from flask import Flask,request,jsonify, render_template
from inference import get_model
from config import *
app = Flask(__name__)

model = get_model()


@app.route('/')
def index():
	return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    notes = request.json["notes"]
    try:
        out = model.predict(notes)
        return jsonify({"result": out})
    except Exception as e:
        return jsonify({"result": "Model Failed"})


@app.route("/process", methods=['POST'])
def process():
    notes = request.form["notes"]
    try:
        out = model.predict(notes)

        family_history = list()
        social_history = list()
        original_notes = ""
        for note in out:
            original_notes += note["Original Notes"]
            original_notes += ', '
            if note["Predict Notes"] == "":
                continue
            if 'f' in note["Predict Notes"]:
                family_history += note["Predict Notes"]['f']
            if 's' in note["Predict Notes"]:
                social_history += note["Predict Notes"]['s']

        if not family_history:
            family_history += ["Not Found!"]

        if not social_history:
            social_history += ["Not Found!"]

        return render_template("index.html", original_notes=original_notes,
                               family_history=family_history, social_history=social_history)
    except Exception as e:
        return jsonify({"result": "Model Failed"})


if __name__ == "__main__":
    app.run('0.0.0.0', port=PORT_NUMBER)
