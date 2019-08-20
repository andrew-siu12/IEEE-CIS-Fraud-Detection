import os
import gc
import numpy as np
import pandas as pd
from src.util import *
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from time import time
import datetime
from sklearn.model_selection import TimeSeriesSplit

def base_model(X, y, N_SPLITS):
    """
    Base Model for the dataset. Return the feature importance dataframe
    to identify important features .

    :param X:  dataframe. Containg the training data
    :param Y: dataframe Containg the fraud label
    :param N_SPLITS:  int. the number of  folds
    :return: feature_importances, dataframe that contain feature importance for each fold
    """
    folds = TimeSeriesSplit(n_splits=N_SPLITS)
    model = lgb.LGBMClassifier(random_state=50)
    hyperparameters = model.get_params()
    hyperparameters['metric'] = 'auc'

    aucs = list()
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns

    for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
        start_time = time()
        print(f"Training on fold {fold + 1}")

        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
        clf = lgb.train(hyperparameters, trn_data, 10000, valid_sets=[trn_data, val_data],
                        verbose_eval=1000, early_stopping_rounds=500)

        feature_importances[f'fold_{fold + 1}'] = clf.feature_importance()
        aucs.append(clf.best_score['valid_1']['auc'])

        fold_end_time = datetime.timedelta(seconds=time() - start_time)

        print(f'Fold {fold + 1} finished in {str(fold_end_time)}')

    print('-' * 30)
    print('Train finished')
    print(f'Mean AUC: {np.mean(aucs)}')
    print('-' * 30)

    return feature_importances


def main():
    RAW_DATA_PATH = "../data"
    print("Loading data....")
    train_merge = load_and_merge(RAW_DATA_PATH, 'train')
    test_merge = load_and_merge(RAW_DATA_PATH, 'test')
    print("Finish Loading data....")
    train_merge = reduce_mem_usage(train_merge)
    test_merge = reduce_mem_usage(test_merge)
    print("Reduction of memory success")

    print(f"Merged training set shape: {train_merge.shape}")
    print(f"Merged testing set shape: {test_merge.shape}")

    for col in train_merge.columns:
        if train_merge[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(train_merge[col].values) + list(test_merge[col].values))
            train_merge[col] = le.transform(list(train_merge[col].values))
            test_merge[col] = le.transform(list(test_merge[col].values))

    X_train = train_merge.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_fraud = train_merge.sort_values('TransactionDT')['isFraud']

    feature_importance = base_model(X_train, y_fraud, 5)
    feature_importance['average'] = feature_importance[[f'fold_{fold + 1}' for fold in range(5)]].mean(
        axis=1)
    feature_importance.to_csv('feature_importances.csv')


if __name__ == "__main__":
    main()

