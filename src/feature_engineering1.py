from src.util import *
from src.const import *


def email_feature(train_merge, test_merge):
    """
    Group low  value counts domain as otherss and same domain group together.
    Create  domain suffix feature
    :param train_merge:  pandas dataframe . Merged of transaction and identity dataset
    :param test_merge:  pandas dataframe. Merged of testing ttransaction and identity dataset
    :return:   updated column of train_merge and test)merge
    """
    # train_merge['P_Isproton'] = (train_merge['P_emaildomain'] == 'protonmail.com')
    # train_merge['R_Isproton'] = (train_merge['R_emaildomain'] == 'protonmail.com')
    # test_merge['P_Isproton'] = (test_merge['P_emaildomain'] == 'protonmail.com')
    # test_merge['R_Isproton'] = (test_merge['R_emaildomain'] == 'protonmail.com')

    # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum',
              'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
              'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft',
              'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
              'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
              'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft',
              'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
              'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
              'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
              'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
              'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other',
              'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
              'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other',
              'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other',
              'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',
              'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
              'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']

    for col in ['P_emaildomain', 'R_emaildomain']:
        train_merge[col + '_bin'] = train_merge[col].map(emails)
        test_merge[col + '_bin'] = test_merge[col].map(emails)

        train_merge[col + '_suffix'] = train_merge[col].map(lambda x: str(x).split('.')[-1])
        test_merge[col + '_suffix'] = test_merge[col].map(lambda x: str(x).split('.')[-1])

        train_merge[col + '_suffix'] = train_merge[col + '_suffix'].map(
            lambda x: x if str(x) not in us_emails else 'us')
        test_merge[col + '_suffix'] = test_merge[col + '_suffix'].map(
            lambda x: x if str(x) not in us_emails else 'us')

    return train_merge, test_merge

def transactionamt_feature(train_merge, test_merge):
    """
    add client uID based on Card features and addr columns. Too many unique values for TransactionAmt,
    thus not generalize well. Use aggregations of features to reduce the noise
    :param train_merge:  pandas dataframe . Merged of transaction and identity dataset
    :param test_merge:  pandas dataframe. Merged of testing ttransaction and identity dataset
    :return:   updated column of train_merge and test)merge
    """

    train_merge['uid'] = train_merge['card1'].astype(str) + \
                         '_' + train_merge['card2'].astype(str) + \
                         '_' + train_merge['card3'].astype(str) + \
                         '_' + train_merge['card4'].astype(str)
    test_merge['uid'] = test_merge['card1'].astype(str) + \
                        '_' + test_merge['card2'].astype(str) + \
                        '_' + test_merge['card3'].astype(str) + \
                        '_' + test_merge['card4'].astype(str)

    train_merge['uid2'] = train_merge['uid'].astype(str) + \
                          '_' + train_merge['addr1'].astype(str) + '_' + \
                          train_merge['addr2'].astype(str)
    test_merge['uid2'] = test_merge['uid'].astype(str) + \
                         '_' + test_merge['addr1'].astype(str) + '_' + \
                         test_merge['addr2'].astype(str)
    card_uid_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2']

    for col in card_uid_cols:
        for agg_type in ['mean', 'std']:
            new_col_name = col + '_TransactionAmt_' + agg_type
            temp_df = pd.concat([train_merge[[col, 'TransactionAmt']],
                                 test_merge[[col, 'TransactionAmt']]])
            temp_df = temp_df.groupby([col])['TransactionAmt'].agg(
                [agg_type]).reset_index().rename(columns={agg_type: new_col_name})
            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()
            train_merge[new_col_name] = train_merge[col].map(temp_df)
            test_merge[new_col_name] = test_merge[col].map(temp_df)
    return train_merge, test_merge

def m_cols_features(train_merge, test_merge):
    """
        Create  M_sum columns to sum across m-cols  and craete column to calculate nulls across m columns
        :param train_merge:  pandas dataframe . Merged of transaction and identity dataset
        :param test_merge:  pandas dataframe. Merged of testing ttransaction and identity dataset
        :return:   updated column of train_merge and test)merge
        """
    m_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
    train_merge['M_sum'] = train_merge[m_cols].sum(axis=1).astype(np.int8)
    test_merge['M_sum'] = test_merge[m_cols].sum(axis=1).astype(np.int8)

    train_merge['M_nulls'] = train_merge[m_cols].isna().sum(axis=1).astype(np.int8)
    test_merge['M_nulls'] = test_merge[m_cols].isna().sum(axis=1).astype(np.int8)
    return train_merge, test_merge

def c_features(train_merge, test_merge):
    """
        create feature to identify whether the transaction belongs to common value counts of C features
        :param train_merge:  pandas dataframe . Merged of transaction and identity dataset
        :param test_merge:  pandas dataframe. Merged of testing ttransaction and identity dataset
        :return:   updated column of train_merge and test)merge
        """
    c_cols = train_merge.iloc[:, 17:31].columns

    train_merge['C_sum'] = 0
    test_merge['C_sum'] = 0

    train_merge['C_null'] = 0
    test_merge['C_null'] = 0

    for col in c_cols:
        train_merge['C_sum'] += np.where(train_merge[col] == 1, 1, 0)
        test_merge['C_sum'] += np.where(test_merge[col] == 1, 1, 0)

        train_merge['C_null'] += np.where(train_merge[col] == 0, 1, 0)
        test_merge['C_null'] += np.where(test_merge[col] == 0, 1, 0)

        valid_values = train_merge[col].value_counts()
        valid_values = valid_values[valid_values > 1000]
        valid_values = list(valid_values.index)

        train_merge[col + '_valid'] = np.where(train_merge[col].isin(valid_values), 1, 0)
        test_merge[col + '_valid'] = np.where(test_merge[col].isin(valid_values), 1, 0)
    return train_merge, test_merge

def main():
    RAW_DATA_PATH = "../preprocessed_data"
    print("Loading and merging data")
    train_merge = load_and_merge(RAW_DATA_PATH, 'train')
    test_merge = load_and_merge(RAW_DATA_PATH, 'test')

    train_merge = reduce_mem_usage(train_merge)
    test_merge = reduce_mem_usage(test_merge)
    print(f"Merged training set shape: {train_merge.shape}")
    print(f"Merged testing set shape: {test_merge.shape}")

   # cols_to_drop = get_cols_to_drop(train_merge, BASE_COLUMNS)
   # print(f"Columns to drop : \n {cols_to_drop}")

    # make a feature that contains the total null values for each transaction
    train_merge['nulls_count'] = train_merge.isna().sum(axis=1)
    test_merge['nulls_count'] = test_merge.isna().sum(axis=1)

    # Extract day and hour features from TransactionDT
    train_merge['Transaction_day'] = make_day_feature(train_merge)
    test_merge['Transaction_day'] = make_day_feature(test_merge)
    train_merge['Transaction_hour'] = make_hour_feature(train_merge)
    test_merge['Transaction_hour'] = make_hour_feature(test_merge)

    train_merge, test_merge = email_feature(train_merge, test_merge)
    train_merge, test_merge = transactionamt_feature(train_merge, test_merge)
    train_merge, test_merge = m_cols_features(train_merge, test_merge)
    train_merge, test_merge = c_features(train_merge, test_merge)

    freq_cols = ['card1', 'card2', 'card3', 'card5',
                 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
                 'addr1', 'addr2',
                 'dist1', 'dist2',
                 'P_emaildomain', 'R_emaildomain',
                 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10',
                 'id_11', 'id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_24',
                 'id_25', 'id_26', 'id_30', 'id_31', 'id_32', 'id_33',
                 'DeviceInfo', 'DeviceInfo_c', 'id_30_c', 'id_30_v', 'id_31_v',
                 ]

    for col in freq_cols:
        temp_df = pd.concat([train_merge[[col]], test_merge[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
        train_merge[col + '_fq_enc'] = train_merge[col].map(fq_encode)
        test_merge[col + '_fq_enc'] = test_merge[col].map(fq_encode)

    # Label Encode object columns
    for col in train_merge.columns:
        if train_merge[col].dtype == 'O':
            train_merge[col] = train_merge[col].astype(str)
            test_merge[col] = test_merge[col].astype(str)

            le = LabelEncoder()
            le.fit(list(train_merge[col]) + list(test_merge[col]))
            train_merge[col] = le.transform(train_merge[col])
            test_merge[col] = le.transform(test_merge[col])

            train_merge[col] = train_merge[col].astype('category')
            test_merge[col] = test_merge[col].astype('category')

    features_check = []
    columns_to_check = set(list(train_merge)).difference(base_columns)
    for col in columns_to_check:
        features_check.append(ks_2samp(test_merge[col], train_merge[col])[1])

    features_check = pd.Series(features_check, index=columns_to_check).sort_values()
    features_discard = list(features_check[features_check == 0].index)
    features_discard = features_discard + COLS_TO_DROP
    print(f"The number of features to be removed {len(features_discard)}")

    train_merge = train_merge.drop(features_discard, axis=1)
    test_merge = test_merge.drop(features_discard, axis=1)

    print(f"The shape of final transformed training data {train_merge.shape}")
    print(f"The shape of final transformed testing data {test_merge.shape}")



if __name__ == '__main__':
    main()