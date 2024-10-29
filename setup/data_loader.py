if __name__ == "__main__":
    import os
    import sys

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)

import pandas as pd
from database import db, User, PaymentData
from constants import SAMPLEDATA_USER_FILE, SAMPLEDATA_PAYMENT_FILE


def run():
    """
    Loads the sample data from the user and payment files into the database.
    """
    # Create tables
    db.connect()
    print(f"Dropping all tables")
    db.drop_tables([User, PaymentData])
    print(f"Re-creating tables")
    db.create_tables([User, PaymentData])

    user_df = pd.read_csv(SAMPLEDATA_USER_FILE)
    payments_df = pd.read_csv(SAMPLEDATA_PAYMENT_FILE)
    user_records = user_df.to_dict(orient="records")
    payment_records = payments_df.to_dict(orient="records")
    with db.atomic():  # Ensures all inserts are done in a single transaction
        print("Inserting sample user data into DB")
        User.insert_many(user_records).execute()

        print("Inserting sample payments data into DB")
        PaymentData.insert_many(payment_records).execute()

        print("Done!")


if __name__ == "__main__":
    run()
