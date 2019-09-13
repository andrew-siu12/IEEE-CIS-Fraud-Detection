import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import tpe, Trials, STATUS_OK, hp, fmin
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def objective(hyperparameters):
    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']

    # Extract the boosting type and subsample to top levl keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'min_child_samples', 'max_depth']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    training_start = time()

    print("New RUN")
    FOLDS = 5

    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    mean_score = 0

    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        start_time = time()

        trn_x, trn_y = X_train.iloc[tr_idx, :], y_train.iloc[tr_idx, :]
        val_x, val_y = X_train.iloc[val_idx, :], y_train.iloc[val_idx, :]

        trn_data = lgb.Dataset(trn_x, label=trn_y)
        val_data = lgb.Dataset(val_x, label=val_y)

        clf = lgb.train(hyperparameters, trn_data, 10000, valid_sets=[trn_data, val_data],
                        verbose_eval=1000, early_stopping_rounds=500)

        y_pred_valid = clf.predict(val_x)
        val_roc_auc = roc_auc_score(val_y, y_pred_valid)
        mean_score += val_roc_auc / FOLDS

        fold_finish_time = datetime.timedelta(seconds=time() - start_time)
        print(f"Fold {fold + 1} auc_score: {val_roc_auc} and finished in {fold_finish_time}")

    total_time = time() - training_start
    print(f"Total tuning time: {round(total_time / 60, 2)}")
    gc.collect()

    print(f"Mean AUC score = {mean_score}")
    del trn_x, trn_y, val_x, val_y

    # write to the csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([mean_score, hyperparameters, total_time])
    of_connection.close()

    return {'loss': -mean_score, 'hyperparameters': hyperparameters,
            'train_time': total_time, 'status': STATUS_OK}


def main():

    with open('../preprocessed_data/data_type_key_test.json') as data_file:
        data_types_train = json.load(data_file)

    with open('../preprocessed_data/data_type_key_test.json') as data_file:
        data_types_test = json.load(data_file)

    X_train = pd.read_csv("../preprocessed_data/X_train.csv", dtype=data_types_train)
    y_train = pd.read_csv("../preprocessed_data/y_train.csv", header=None)
    X_test = pd.read_csv("../preprocessed_data/X_test.csv", dtype=data_types_test)

    space = {
        'boosting_type': hp.choice('boosting_type',
                                   [{'boosting_type': 'gbdt'},
                                    {'boosting_type': 'goss'}]),
        'num_leaves': hp.quniform('num_leaves', 100, 300, 10),
        'max_depth': hp.quniform('max_depth', 7, 20, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'min_child_samples': hp.quniform('min_child_samples', 100, 250, 10),
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.4),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.4),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.4, 0.9),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 0.7),
        'feature_fraction': hp.uniform('feature_fraction', 0.2, 0.7),
        'objective': 'binary',
        'metric': 'auc',
        'tree_learner': 'serial',
        "bagging_seed": 11,
    }

    OUT_FILE = '../models/tuning.csv'
    of_connection = open(OUT_FILE, 'w')
    writer = csv.writer(of_connection)

    # Write column names
    headers = ['loss', 'hyperparameters', 'runtime']
    writer.writerow(headers)
    of_connection.close()

    MAX_EVALS = 10
    best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=Trials(), max_evals=MAX_EVALS)
    best_params = space_eval(space, best)
    # Extract the boosting type and subsample to top levl keys
    # subsample = best_params['boosting_type']['subsample']
    best_params['boosting_type'] = best_params['boosting_type']['boosting_type']

    print(best_params)

if __name__ == "__main__":

