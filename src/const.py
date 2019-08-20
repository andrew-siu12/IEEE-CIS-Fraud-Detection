import pandas as pd

train_identity = pd.read_csv("data/train_identity.csv", )
train_transaction = pd.read_csv("data/train_transaction.csv", )
test_identity = pd.read_csv("data/test_identity.csv", )
test_transaction = pd.read_csv("data/test_transaction.csv",)


BASE_COLUMNS  = list(train_transaction.columns) + list(train_identity.columns)