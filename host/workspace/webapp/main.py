import datetime
import pymongo

from flask import Flask
from flask_pymongo import PyMongo

from flask import request, render_template, jsonify

import json

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/intelliTrain"
mongo = PyMongo(app)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/create", methods=['POST'])
def create_evaluation():
    """ Insert evaluation results to database
    Use the following command to test POST:
        curl --header "Content-Type: application/json" --request POST --data @set-00-October-21-2019--04:18PM.json http://localhost/create
    """
    # format as document:
    evaluation = request.get_json()
    evaluation = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), 
        "evaluation": evaluation
    }
    # insert:
    mongo.db.evaluations.insert_one(evaluation).inserted_id
    return ('', 204)

@app.route("/read")
def read_evaluation():
    # find the latest:
    evaluations = mongo.db.evaluations.find().sort(
        [
            ("timestamp", pymongo.DESCENDING)
        ]
    ).limit(1)

    response = []
    for evaluation in evaluations:
        response.append(
            {
                "data": [
                    [
                        int(round(e["timestamp"])), int(round(e["score"]))
                    ] for e in evaluation["evaluation"]
                ],
                "type": "line"
            }
        )
    
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=80)