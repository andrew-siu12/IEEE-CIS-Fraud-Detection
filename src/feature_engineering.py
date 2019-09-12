import os
import gc
import numpy as np
import pandas as pd
from src.util import *
from sklearn.preprocessing import LabelEncoder
from time import time
import datetime
from scipy.stats import ks_2samp


def id_split(train_identity, test_identity):
    """
    Group same mobile phone company with different build into same group, lower count of device group into others
    Seperate the mobile phone device name and the version number into seperate features.
    Seperate the mobile phone OS and version number into sperate features
    Seperate the browser OS and version number into sperate feautures
    :param train_identity: dataframe:
    :param test_identity:  dataframe
    :return:  updated column of train_identity and test_identity
    """
    for df in [train_identity, test_identity]:
        df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
        df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]

        df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
        df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

        df['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
        df['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]

        df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
        df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]

        df['id_34'] = df['id_34'].str.split(':', expand=True)[1]
        df['id_23'] = df['id_23'].str.split(':', expand=True)[1]

        df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
        df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
        df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
        df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
        df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
        df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
        df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
        df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
        df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
        df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
        df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
        df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
        df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
        df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
        df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
        df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
        df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

        df.loc[df.device_name.isin(df.device_name.value_counts()[
                                       df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
        df['had_id'] = 1
        gc.collect()

    return train_identity, test_identity


def main():
    RAW_DATA_PATH = "../data"
    train_identity = pd.read_csv(RAW_DATA_PATH + "/train_identity.csv")
    test_identity = pd.read_csv(RAW_DATA_PATH + "/test_identity.csv")
    print("Finishing Loading data....")
    print(train_identity.shape, test_identity.shape)

    train_identity, test_identity = id_split(train_identity, test_identity)

    print("feature transformation has finished....")

    print(train_identity.shape)
    print()
    print(test_identity.shape)

    OUT_PATH = "../preprocessed_data"
    train_identity.to_csv(OUT_PATH + "/train_identity.csv", index=False)
    test_identity.to_csv(OUT_PATH + "/test_identity.csv", index=False)


if __name__ == "__main__":
    main()