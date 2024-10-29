import pandas as pd


def convert_userdata_to_df(user_data: dict) -> dict:
    """
    Takes data in the format
    {
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
    and converts it into the format
    [{
        "USER_ID": 22341,
        "CREDIT_LIMIT": 250000.0,
        "SEX": 2,
        "EDUCATION": 1,
        "MARITALSTATUS": 2,
        "AGE": 25,
        "PAYMENTDELAY_1": 0,
        ...
        "BILL_AMT1": 18761.0,
        ...
        "PAID_AMT1": 4662.0,
        ...
    }]
    """
    df = pd.DataFrame(user_data["PAYMENT_DATA"])

    # Create pivot tables for each metric
    delays = df.pivot(columns="MONTH", values="PAYMENTDELAY").add_prefix(
        "PAYMENTDELAY_"
    )
    bills = df.pivot(columns="MONTH", values="BILL_AMT").add_prefix("BILL_AMT")
    payments = df.pivot(columns="MONTH", values="PAID_AMT").add_prefix("PAID_AMT")

    # Convert to float
    bills = bills.astype(float)
    payments = payments.astype(float)

    # Create the base dictionary with ordered fields
    result = pd.Series(
        {
            "USER_ID": user_data["USER_ID"],
            "CREDIT_LIMIT": float(user_data["CREDIT_LIMIT"]),
            "SEX": user_data["SEX"],
            "EDUCATION": user_data["EDUCATION"],
            "MARITALSTATUS": user_data["MARITALSTATUS"],
            "AGE": user_data["AGE"],
        }
    )

    # Combine all series in order
    final_series = pd.concat(
        [
            result,
            delays.iloc[0],
            bills.iloc[0],
            payments.iloc[0],
        ]
    )

    # Convert to dictionary maintaining order
    return final_series.to_dict()
