# Create a flask app
from flask import Flask, jsonify
from flask_cors import CORS
from service import get_all_users, predict, get_user_details

app = Flask(__name__)

CORS(app, supports_credentials=True)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/")
def hello_world():
    return "Hello, World!"


# Returns users in the database
@app.route("/users")
def get_users():
    users = get_all_users()
    return jsonify(users)


# Returns demographic and payment details of one user
@app.route("/users/<int:user_id>")
def get_userdetails(user_id):
    user_details = get_user_details(user_id)
    return jsonify(user_details)


# Returns the default prediction and prediction
# confidence of one user
@app.route("/users/<int:user_id>/prediction")
def predict_(user_id):
    prediction = predict(user_id)
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True)
