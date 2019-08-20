import os
import gc
import numpy as np
import pandas as pd
from src.util import *
from sklearn.preprocessing import LabelEncoder
from time import time
import datetime
from scipy.stats import ks_2samp


def device_features(train_identity, test_identity):
    """
    Group same mobile phone company with different build into same group,
    fill missing values with unknown device
    :param train_identity: dataframe:
    :param test_identity:  dataframe
    :return:  updated column of train_identity and test_identity
    """
    train_identity['DeviceInfo'] = train_identity['DeviceInfo'].fillna(
        'unknown_device').str.lower()
    test_identity['DeviceInfo'] = test_identity['DeviceInfo'].fillna(
        'unknown_device').str.lower()

    train_identity['DeviceInfo_c'] = train_identity['DeviceInfo']
    test_identity['DeviceInfo_c'] = test_identity['DeviceInfo']

    device_match_dict = {
        'sm': 'sm-',
        'sm': 'samsung',
        'huawei': 'huawei',
        'moto': 'moto',
        'rv': 'rv:',
        'trident': 'trident',
        'lg': 'lg-',
        'htc': 'htc',
        'blade': 'blade',
        'windows': 'windows',
        'lenovo': 'lenovo',
        'linux': 'linux',
        'f3': 'f3',
        'f5': 'f5'
    }

    for dev_type_s, dev_type_o in device_match_dict.items():
        train_identity['DeviceInfo_c'] = train_identity['DeviceInfo_c'].apply(
            lambda x: dev_type_s if dev_type_o in x else x)
        test_identity['DeviceInfo_c'] = test_identity['DeviceInfo_c'].apply(
            lambda x: dev_type_s if dev_type_o in x else x)

    # types that are not  in device_match_dict are assigned other_d_type
    train_identity['DeviceInfo_c'] = train_identity['DeviceInfo_c'].apply(
        lambda x: 'other_d_type' if x not in device_match_dict else x)
    test_identity['DeviceInfo_c'] = test_identity['DeviceInfo_c'].apply(
        lambda x: 'other_d_type' if x not in device_match_dict else x)

    return train_identity, test_identity


def id_30_feature(train_identity, test_identity):
     """
    Group same desktop os company with different build into same group
    fCreate features that contain the version number of system.
    fill missing values with unknown device

    :param train_identity: dataframe:
    :param test_identity:  dataframe
    :return:  updated column of train_identity and test_identity
    """
     train_identity['id_30'] = train_identity['id_30'].fillna(
         'unknown_device').str.lower()
     test_identity['id_30'] = test_identity['id_30'].fillna(
         'unknown_device').str.lower()

     train_identity['id_30_c'] = train_identity['id_30']
     test_identity['id_30_c'] = test_identity['id_30']

     device_match_dict = {
         'ios': 'ios',
         'windows': 'windows',
         'mac': 'mac',
         'android': 'android'
     }

     for dev_type_s, dev_type_o in device_match_dict.items():
         train_identity['id_30_c'] = train_identity['id_30_c'].apply(
             lambda x: dev_type_s if dev_type_o in x else x)
         test_identity['id_30_c'] = test_identity['id_30_c'].apply(
             lambda x: dev_type_s if dev_type_o in x else x)

     train_identity['id_30_v'] = train_identity['id_30'].apply(
         lambda x: ''.join([i for i in x if i.isdigit()]))
     test_identity['id_30_v'] = test_identity['id_30'].apply(
         lambda x: ''.join([i for i in x if i.isdigit()]))

     train_identity['id_30_v'] = np.where(
         train_identity['id_30_v'] != '', train_identity['id_30_v'], 0).astype(int)
     test_identity['id_30_v'] = np.where(
         test_identity['id_30_v'] != '', test_identity['id_30_v'], 0).astype(int)

     return train_identity, test_identity


def id_31_feature(train_identity, test_identity):
    """
       make browser build and build number as seperate feature
       fill missing values with unknown br

       :param train_identity: dataframe:
       :param test_identity:  dataframe
       :return:  updated column of train_identity and test_identity
       """
    train_identity['id_31'] = train_identity['id_31'].fillna(
        'unknown_br').str.lower()
    test_identity['id_31'] = test_identity['id_31'].fillna(
        'unknown_br').str.lower()

    train_identity['id_31'] = train_identity['id_31'].apply(
        lambda x: x.replace('webview', 'webvw'))
    test_identity['id_31'] = test_identity['id_31'].apply(
        lambda x: x.replace('webview', 'webvw'))

    train_identity['id_31'] = train_identity['id_31'].apply(
        lambda x: x.replace('for', ' '))
    test_identity['id_31'] = test_identity['id_31'].apply(
        lambda x: x.replace('for', ' '))

    browser_list = set(
        list(train_identity['id_31'].unique()) + list(test_identity['id_31'].unique()))
    browser_list2 = []
    for item in browser_list:
        browser_list2 += item.split(' ')
    browser_list2 = list(set(browser_list2))

    browser_list3 = []
    for item in browser_list2:
        browser_list3 += item.split('/')
    browser_list3 = list(set(browser_list3))

    for item in browser_list3:
        train_identity['id_31_e_' + item] = np.where(
            train_identity['id_31'].str.contains(item), 1, 0).astype(np.int8)
        test_identity['id_31_e_' + item] = np.where(
            test_identity['id_31'].str.contains(item), 1, 0).astype(np.int8)
        if train_identity['id_31_e_' + item].sum() < 100:
            del train_identity['id_31_e_' + item], test_identity['id_31_e_' + item]

    train_identity['id_31_v'] = train_identity['id_31'].apply(
        lambda x: ''.join([i for i in x if i.isdigit()]))
    test_identity['id_31_v'] = test_identity['id_31'].apply(
        lambda x: ''.join([i for i in x if i.isdigit()]))

    train_identity['id_31_v'] = np.where(
        train_identity['id_31_v'] != '', train_identity['id_31_v'], 0).astype(int)
    test_identity['id_31_v'] = np.where(
        test_identity['id_31_v'] != '', test_identity['id_31_v'], 0).astype(int)

    return train_identity, test_identity


def main():
    RAW_DATA_PATH = "../data"
    train_identity = pd.read_csv(RAW_DATA_PATH + "/train_identity.csv")
    test_identity = pd.read_csv(RAW_DATA_PATH + "/test_identity.csv")
    print("Finishing Loading data....")

    train_identity, test_identity = device_features(train_identity, test_identity)
    train_identity, test_identity = id_30_feature(train_identity, test_identity)
    train_identity, test_identity = id_31_feature(train_identity, test_identity)

    print("feature transformation has finished....")

    print(train_identity.columns)
    print()
    print(test_identity.columns)

    train_identity.to_csv("../preprocessed_data/train_identity.csv", index=False)
    test_identity.to_csv("../preprocessed_data/test_identity.csv", index=False)


if __name__ == "__main__":
    main()