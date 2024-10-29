# This is the core file with our business logic

from peewee import fn
import joblib
import pandas as pd
from database import User
from utils import convert_userdata_to_df
from constants import SAMPLE_MODELINPUT


def predict_one(user_data: dict) -> dict:
    """
    Predicts the loan risk for a single user.

    Args:
        user_data (dict): A dictionary containing the features of the loan application.

    Returns:
        dict: A dictionary containing the predicted class and its associated confidence.

    Example:
        {
            "PREDICTION": 1,
            "CONFIDENCE": 0.95
        }
    """
    # 1. Load the data into a dataframe
    df = pd.DataFrame([user_data])
    # 2. Load our loan classification model
    model = joblib.load("risk_radar_model")
    prediction = model.predict(df).item()
    probs = model.predict_proba(df)
    confidence = probs.max(axis=1).item()
    # 3. Return the predictions
    return {"PREDICTION": prediction, "CONFIDENCE": confidence}


def get_all_users() -> list[dict]:
    """
    Retrieve a limited set of users from the database

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing user data. Each dictionary
            has the following keys:
                - "USER_ID" (int): The unique identifier for the user.
                - "USER_NAME" (str): The username of the user.
    Example:
        [{
            "USER_ID": 22341,
            "USER_NAME": "Sandra Lewis"
        }]
    """
    results = (
        User.select(User.USER_ID, User.USER_NAME).order_by(fn.Rand()).limit(20).dicts()
    )
    results = [dict(row) for row in results]
    return results


def get_user_details(user_id) -> dict:
    """
    Retrieves all details for a specific user from the database.

    Args:
        user_id (int): The unique identifier of the user to retrieve details for.

    Returns:
        Dict[str, Any]: A dictionary containing the user's demograpphic and payment details..

    Example:
        {
            "USER_ID": 24,
            "USER_NAME": 24,
            "AGE": 24,
            "CREDIT_LIMIT": 20000,
            "DID_DEFAULT_PAYMENT": 1,
            "EDUCATION": 2,
            "MARITALSTATUS": 1,
            "PAYMENT_DATA": [
                {
                "BILL_AMT": 3913,
                "MONTH": 1,
                "PAID_AMT": 0,
                "PAYMENTDELAY": 2
                },
                ...
            ],
            "SEX": 2,
            "USER_ID": 1,
            "USER_NAME": "Sandra Lewis"
        }
    """
    user = User.get(User.USER_ID == user_id)
    user_dict = {
        "USER_ID": user.USER_ID,
        "USER_NAME": user.USER_NAME,
        "CREDIT_LIMIT": user.CREDIT_LIMIT,
        "SEX": user.SEX,
        "EDUCATION": user.EDUCATION,
        "MARITALSTATUS": user.MARITALSTATUS,
        "AGE": user.AGE,
        "DID_DEFAULT_PAYMENT": user.DID_DEFAULT_PAYMENT,
        "PAYMENT_DATA": [
            {
                "MONTH": payment.MONTH,
                "PAYMENTDELAY": payment.PAYMENTDELAY,
                "BILL_AMT": payment.BILL_AMT,
                "PAID_AMT": payment.PAID_AMT,
            }
            for payment in user.payments
        ],
    }
    return user_dict


def predict(user_id: int) -> dict:
    """
    Predicts the payment default risk for a single user.

    Args:
        user_id (int): The id of the user to predict for.

    Returns:
        dict: A dictionary containing the predicted class and its associated confidence.

    Example:
        {
            "PREDICTION": 1,
            "CONFIDENCE": 0.95
        }
    """
    user_details = get_user_details(user_id)
    user_data = convert_userdata_to_df(user_details)
    prediction = predict_one(user_data)
    return prediction


if __name__ == "__main__":
    print("********")
    print("    Getting prediction for User: 22341 by passing the full data")
    print("********")

    test_data = SAMPLE_MODELINPUT
    prediction = predict_one(test_data)
    print(prediction)

    print("\n\n")

    print("********")
    print("    Getting all users")
    print("********")

    users = get_all_users()
    print(users)

    print("\n\n")

    print("********")

    user_id_to_predict = 3
    print(
        f"    Getting prediction for  User: {user_id_to_predict} by using the user_id"
    )
    print("********")

    users = predict(user_id_to_predict)
    print(users)
