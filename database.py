# All our database related operations go here

from enum import Enum
from peewee import (
    MySQLDatabase,
    Model,
    FloatField,
    IntegerField,
    CharField,
    ForeignKeyField,
)

db = MySQLDatabase(
    "defaultdb",
    user="<your_db_username>",
    password="<your_db_password>",
    host="<your_db_host>",
    port=3306,
)


class Sex(Enum):
    MALE = 1
    FEMALE = 2


class Education(Enum):
    GRADUATE = 1
    UNIVERSITY = 2
    HIGH_SCHOOL = 3
    OTHERS = 4


class MaritalStatus(Enum):
    MARRIED = 1
    SINGLE = 2
    OTHERS = 3


class PaymentDefault(Enum):
    YES = 1
    NO = 0


class User(Model):
    USER_ID = IntegerField(primary_key=True)
    USER_NAME = CharField()
    CREDIT_LIMIT = FloatField()
    SEX = IntegerField(choices=[(status.value, status.name) for status in Sex])
    EDUCATION = IntegerField(
        choices=[(status.value, status.name) for status in Education]
    )
    MARITALSTATUS = IntegerField(
        choices=[(status.value, status.name) for status in MaritalStatus]
    )
    AGE = IntegerField()
    DID_DEFAULT_PAYMENT = IntegerField(
        choices=[(status.value, status.name) for status in PaymentDefault]
    )

    class Meta:
        database = db


class PaymentData(Model):
    # foreign key to User
    USER_ID = ForeignKeyField(User, backref="payments", on_delete="CASCADE")
    MONTH = IntegerField()
    PAYMENTDELAY = IntegerField()
    BILL_AMT = FloatField()
    PAID_AMT = FloatField()

    class Meta:
        database = db
