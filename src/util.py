import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def resumetable(df):
    """
    Return summary sof the dataframe e.g. Missing values, unique values, first few values of the dataset

    :param
        df:  pandas dataframe
    :return:
        summary: dataframe that contains summary of  df
    """
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    return summary

# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
def count_fraud_plot(df, col):
    """
       Return  a count  distribution plot against categorical column  and the fraud percentage of each category
       :param df:  pandas dataframe
       :param col_name:  str . categorical column
       :return:
       """

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
    sns.countplot(x=col, data=df,
                  order=list(tmp[col].values), ax=ax[0])

    # ax[0].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[0].set_title(
        f"Frequency of {col} values", fontsize=16)
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

    for p in ax[0].patches:
        height = p.get_height()
        ax[0].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height / df.shape[0] * 100:.2f}%',
                   ha='center', fontsize=8)

    sns.barplot(x=col, y='Fraud', data=tmp,
                order=list(tmp[col].values), ax=ax[1])

    ax[1].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[1].set_title(
        f"Fraud Percentage of {col} values", fontsize=16)
    ax[1].set_ylabel("Percent", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    plt.subplots_adjust(hspace=.4, top=0.9)

    for p in ax[1].patches:
        height = p.get_height()
        ax[1].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height:.2f}%',
                   ha='center', fontsize=8)
    plt.subplots_adjust(hspace=.4, top=1.1)
    plt.show();


def amt_fraud_plot(df, col):
    """
          Return  a bar distribution plot  of transaction amount against  categorical col  and the fraud percentage of
          each value of the column
          :param df:  pandas dataframe
          :param col:  str . categorical column
          :return:
          """
    percent_amt = (df.groupby(['isFraud', col])['TransactionAmt'].sum()
                   / df.groupby([col])['TransactionAmt'].sum() * 100). \
        unstack('isFraud')
    percent_amt = percent_amt.reset_index()
    percent_amt.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()

    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
    sns.barplot(x=col, y='TransactionAmt',
                data=amt,
                order=list(percent_amt[col].values),
                ax=ax[0])

    # ax[0].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[0].set_title(
        f"{col} Transaction amounts", fontsize=16)
    ax[0].set_ylabel("Transaction Amount (USD)", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    total_amt = df['TransactionAmt'].sum()
    for p in ax[0].patches:
        height = p.get_height()
        ax[0].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height / total_amt * 100:.2f}%',
                   ha='center', fontsize=8)

    sns.barplot(x=col, y='Fraud', data=percent_amt,
                order=list(percent_amt[col].values), ax=ax[1])

    ax[1].set_xlabel(f"{col} Category Names", fontsize=12)
    ax[1].set_title(
        f"Fraud Percentage of {col} values", fontsize=16)
    ax[1].set_ylabel("Percent", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    plt.subplots_adjust(hspace=.4, top=0.9)

    for p in ax[1].patches:
        height = p.get_height()
        ax[1].text(p.get_x() + p.get_width() / 2.,
                   height,
                   f'{height:.2f}%',
                   ha='center', fontsize=8);
    plt.show();


def group_email_domain(df):
    """
    Function that group  email domain bty the service provider  grouped  email domain that
    has very few coiunts into  category 'other'
    :param df:  pandas dataframe
    :return:  df:  pandas dataframe
    """
    for email in ['P_emaildomain', 'R_emaildomain']:
        df.loc[df[email].
                   isin(['gmail.com', 'gmail']), email] = 'Google Mail'
        df.loc[df[email].
                   isin(['yahoo.com', 'ymail.com', 'yahoo.com.mx',
                         'yahoo.co.jp', 'yahoo.fr', 'yahoo.co.uk',
                         'yahoo.es', 'yahoo.de']), email] = 'Yahoo Mail'
        df.loc[df[email].
                   isin(['hotmail.com', 'outlook.com', 'msn.com',
                         'live.com', 'live.com.mx', 'outlook.es',
                         'hotmail.fr', 'hotmail.co.uk', 'live.fr',
                         'hotmail.es', 'hotmail.de']), email] = 'Microsoft mail'
        df.loc[df[email].
                   isin(['icloud.com', 'me.com', 'mac.com']), email] = 'Apple mail'

        df.loc[df[email].
                   isin(df[email].
                        value_counts()[df[email].
                        value_counts() <= 1000].index), email] = 'Others'
    return df


def plot_cat_features(df, col, lim=2000):
    """
    Extension of  count_fraud_plot  with addition of  box plot 
    :param df:  pandas dataframe
    :param col:  categorical colnum
    :param lim: int that limit the transaction amt
    :return:
    """

    plt.figure(figsize=(14, 10))
    plt.subplot(221)
    g = sns.countplot(x=col, data=df)
    g.set_title(col + ' Distribution', fontsize=14)
    # g.set_xlabel(col + ' Name', fontsize=10)
    g.set_ylabel('Count', fontsize=14)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x() + p.get_width() / 2.,
               height,
               f'{height / df.shape[0] * 100:.2f}%',
               ha='center', fontsize=10)

    plt.subplot(222)
    g1 = df.groupby(col)['isFraud'].mean() \
        .sort_index() \
        .plot(kind='bar',
              title='Percentage of Fraud by ' + col,
              color=sns.color_palette())
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x() + p.get_width() / 2.,
                height,
                f'{height * 100:.2f}%',
                ha='center', fontsize=10)

    plt.subplot(212)
    g2 = sns.boxenplot(x=col, y='TransactionAmt', hue='isFraud',
                       data=df[df['TransactionAmt'] <= lim])
    g2.set_title(f"{col} boxplot by TransactionAmt and Fraud")
    g2.set_ylabel("Transaction Values in $")

    plt.subplots_adjust(hspace=0.5)
    plt.show()