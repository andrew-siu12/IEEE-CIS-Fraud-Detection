# IEEE-CIS-Fraud-Detection

This dataset is provided by kaggle for [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection) 

#### -- Project Status: [Active]

### Technologies
* Python
* Pandas, numpy
* LightGBM
* matplotlib, seaborn
* hyperopts, scikit-learn

## Project Description/Objective

According to [PwC 2018 study](https://www.pwc.com/gx/en/economic-crime-survey/pdf/GlobalEconomicCrimeSurvey2016.pdf). Around 49% of organizations surveyed had experienced fraud crime and it is increasing every year. Thus, a reliable fraud prevention system can save customer millions of dallars per year. In this competition, researchers from the IEEE Computational Intelligence Society (IEEE-CIS) want to improve this figure, while also improving the customer experience. With higher accuracy fraud detection, you can get on with your business without the hassle.

In this project, we will build machine learning models on a large-scale dataset provided by Vesta. The dataset contains real-world e-commerce transactions with more than 300 features. For a successful model, it should minimize both false positive and false negative (legit transaction identify as fraud and fraud transaction identify as non-fraud). If successful, the efficency of fraud transcation alerts will be improved and helping hundreds of business reduce their loss on fraud crime. 

## Dataset
The data is broken into two files **identity.csv** and **transaction.csv**, which are joined by TransactionID. Not all transactions have corresponding identity information.

### Categorical Features - Transaction
* ProductCD
* card1 - card6
* addr1, addr2
* P_emaildomain
* R_emaildomain
* M1 - M9

### Categorical Features - Identity
* DeviceType
* DeviceInfo
* id_12 - id_38

Other features are described in the EDA jupyter notebook in notebook folder.

## EDA
Full EDA code are in the EDA jupyter notebook. Here are some of the findings of the analysis.
* only 3.5% oo transactions are fraud
* The timespan of the dataset is around 1-year. The training and test transaction time do not overlap, which suggests training and test set are split by time. Also, there is around a one-month gap in between training and test set.
* Transaction amount is in USD. The distrubtion of this feature is very skewed. So we take the log transform to better view the distrubution. After applying log-transform, the fraud and non-fraud amount becomes normal distribution.
* most of the fraudent amount are generally lower the non-frauddent amount

## Metrics
In this project, we will use AUC-ROC to evaluate the model. ROC is a probability curve and AUC represents degree of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between transactions with fraud and non-fraud

## Process 
* EDA - to find insights about the data and potential features for feature engineering
* feature engineering - aggreagating features, label encoding and NaN processing features
* model and hyperparameter tuning - Use LightGBM for this problem and hyperopts to optimize the hyperparameters

We repeat the above steps to imporve the score.

## File Descriptions
```
├── README.md
├── models
│   ├── tuning1.csv
│   ├── tuning2.csv
│   └── tuning3.csv
├── notebook
│   ├── Base\ Model.ipynb
│   ├── EDA.ipynb
│   ├── Feature\ Engineering.ipynb
│   ├── Hyperparameter\ tuning.ipynb
│   ├── Predictions.ipynb
│   ├── feature_importances1.csv
│   └── util.py
└── src
    ├── basemodel.py
    ├── const.py
    ├── feature_engineering.py
    ├── feature_engineering1.py
    ├── feature_importances.csv
    ├── hyperparameter_tuning.py
    └── util.py
```
* **models** - contains the results and hyperparameter of the tuned models
* **notebook** - `Base Model.ipynb` contains the baseline model for this project. `EDA.ipynb` contains the full EDA code on both transaction and identity dataset. `Feature Engineering.ipynb` shows prototypes of feature engineering columns. `Hyperparameter tuning.ipynb` demostrates the use of hyperopt to tune the hyperparameters. `Predictions.ipynb` shows the result of final tuned model
* **src** - turn all of the notebook folder into a collection of readable scripts

## Results and Discussion
The tuned LightGBM model submission gives us a 94.00 in the public leaderboard. The result can be improved by repeating the process described above.
