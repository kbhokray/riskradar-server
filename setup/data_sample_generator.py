if __name__ == "__main__":
    import os
    import sys

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)

import pandas as pd
from faker import Faker
from database import Sex
from constants import RAW_DATA_FILE, SAMPLEDATA_USER_FILE, SAMPLEDATA_PAYMENT_FILE

fake = Faker()


def generate_name(sex):
    if sex == Sex.MALE:
        return fake.name_male()
    elif sex == Sex.FEMALE:
        return fake.name_female()
    else:
        return fake.name_nonbinary()


def run():
    """
    Generates fake data and writes it to a user and payment files.
    """

    print(f"Reading data from CSV: {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE)

    user_data_df = df[
        [
            "USER_ID",
            "CREDIT_LIMIT",
            "SEX",
            "EDUCATION",
            "MARITALSTATUS",
            "AGE",
            "DID_DEFAULT_PAYMENT",
        ]
    ].drop_duplicates()

    user_data_df["USER_NAME"] = user_data_df["SEX"].apply(generate_name)

    melted_df = pd.melt(
        df,
        id_vars=["USER_ID"],
        value_vars=[
            "PAYMENTDELAY_1",
            "PAYMENTDELAY_2",
            "PAYMENTDELAY_3",
            "PAYMENTDELAY_4",
            "PAYMENTDELAY_5",
            "PAYMENTDELAY_6",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAID_AMT1",
            "PAID_AMT2",
            "PAID_AMT3",
            "PAID_AMT4",
            "PAID_AMT5",
            "PAID_AMT6",
        ],
        var_name="Month_Column",
        value_name="Value",
    )

    melted_df["MONTH"] = melted_df["Month_Column"].str.extract("(\\d+)$").astype(int)
    melted_df["Type"] = (
        melted_df["Month_Column"]
        .str.extract("([A-Z_]+)")
        .replace(
            {
                "PAYMENTDELAY_": "PAYMENTDELAY",
                "BILL_AMT": "BILL_AMT",
                "PAID_AMT": "PAID_AMT",
            }
        )
    )

    # Pivot the data back so that each month has its own row with relevant columns
    payments_df = melted_df.pivot_table(
        index=["USER_ID", "MONTH"],
        columns="Type",
        values="Value",
        aggfunc="first",
    ).reset_index()

    print(f"Saving Users data to file: {SAMPLEDATA_USER_FILE}")
    user_data_df.to_csv(SAMPLEDATA_USER_FILE, index=False)

    print(f"Saving Payments data to file: {SAMPLEDATA_PAYMENT_FILE}")
    payments_df.to_csv(SAMPLEDATA_PAYMENT_FILE, index=False)

    print("Done!")


if __name__ == "__main__":
    run()
