from sklearn import metrics, model_selection
import joblib
import MachineLearningUtils as mlu
from MachineLearningUtils import utils, ensemble
import numpy as np
import pandas as pd
import os

def classification_single_model_cv(df_labeled, df_target, df_test_encoding, cv_mode, model, list_metrics,
        df_test_id, target_column_name, csv_save_path, fold_num, validation_save_path=None):
    """
    Utilize all labeled data to train & val & test with cross-validation
    Parameters:
        - list_metrics
            - str or list, calculate all the metrics value in validation
            - first metric is used to generate csv filename
        - df_test_id: used to generate csv
        - target_column_name: Y column name of dataframe
    """
    # check value type
    list_metrics = utils.transform_single_value_to_list(list_metrics)

    # pipeline
    array_val_metrics = np.zeros((fold_num, len(list_metrics)))  # initialize
    for fold_index, (train_index, val_index) in enumerate(cv_mode.split(X=df_labeled, y=df_target)):
        X_train = df_labeled.loc[train_index]
        X_val = df_labeled.loc[val_index]
        Y_train = df_target.loc[train_index]
        Y_val = df_target.loc[val_index]

        print("[Debug]: Training...")
        model.fit(X_train, Y_train)

        print("[Debug]: Validation...")
        pred_train = model.predict_proba(X_train)[:, 1]
        pred_val = model.predict_proba(X_val)[:, 1]

        for metric_index, metric in enumerate(list_metrics):
            train_metric = mlu.METRICS_DICT[metric](Y_train, pred_train)
            val_metric = mlu.METRICS_DICT[metric](Y_val, pred_val)
            array_val_metrics[fold_index, metric_index] = val_metric
            print(f"Train_{metric}: {train_metric}, Val_{metric}: {val_metric}")

        print("[Debug]: Saving...")
        # save validation predictions
        if validation_save_path != None:
            pd.DataFrame(pred_val).to_csv(
                    os.path.join(validation_save_path, str(fold_index) + ".csv"), 
                    index=False)
        # save test predictions
        utils.save_kaggle_csv(
                df_test_id, 
                target_column_name, 
                model.predict_proba(df_test_encoding)[:, 1], 
                os.path.join(csv_save_path, 
                        f"{fold_num}_{round(array_val_metrics[fold_index, 0], 4)}_{fold_index}.csv"))

    mean_metrics = np.mean(array_val_metrics, axis=0)
    for index, metric in enumerate(list_metrics):
        print(f"[Debug]: overall cv average {metric}: {mean_metrics[index]}")

def classification_single_model_full(df_labeled, df_target, df_test_encoding, model, list_metrics,
        df_test_id, target_column_name, csv_save_path):
    """
    All the labeled data is used for training
    Parameters:
        - list_metrics
            - str or list, calculate all the metrics value in validation
            - first metric is used to generate csv filename
        - df_test_id: used to generate csv
        - target_column_name: Y column name of dataframe
    """
    # check value type
    list_metrics = utils.transform_single_value_to_list(list_metrics)

    # pipeline
    array_val_metrics = np.zeros(len(list_metrics))  # initialize

    print("[Debug]: Training...")
    model.fit(df_labeled, df_target)

    print("[Debug]: Training Metrics...")
    pred_train = model.predict_proba(df_labeled)[:, 1]
    for metric_index, metric in enumerate(list_metrics):
        train_metric = mlu.METRICS_DICT[metric](df_target, pred_train)
        array_val_metrics[metric_index] = train_metric
        print(f"Train_{metric}: {train_metric}")

    print("[Debug]: Saving...")
    utils.save_kaggle_csv(
            df_test_id, 
            target_column_name, 
            model.predict_proba(df_test_encoding)[:, 1], 
            os.path.join(csv_save_path, 
                    f"{round(array_val_metrics[0], 4)}.csv"))

    for index, metric in enumerate(list_metrics):
        print(f"[Debug]: overall cv average {metric}: {array_val_metrics[index]}")

def classification_single_model_hyperparam_cv(df_labeled, df_target, df_test_encoding, 
        model, tune_mode, list_metrics, search_param, num_cv,
        df_test_id, target_column_name, csv_save_path):
    # check value type
    if not(isinstance(list_metrics, (list, str))):
        print("Error: list_metrics must be list or str type")
        return -1

    print("[Debug]: Training...")
    tune_model = mlu.TUNE_MODEL_DICT[tune_mode](
            estimator=model,
            param_grid=search_param,
            scoring=list_metrics,
            verbose=3,
            n_jobs=-1,
            cv=num_cv)
    tune_model.fit(df_labeled, df_target)

    print("[Debug]: Saving...")
    utils.save_kaggle_csv(
            df_test_id, 
            target_column_name, 
            tune_model.predict_proba(df_test_encoding)[:, 1], 
            os.path.join(csv_save_path, 
                    f"{round(tune_model.best_score_, 4)}.csv"))
                    
    print("[Debug]: Best params: ")
    best_params = tune_model.best_estimator_.get_params()
    for param_name in search_param.keys():
        print(f"{param_name}: {best_params[param_name]}")

def classification_ensemble_cv(df_labeled, df_target, df_test_encoding, 
        df_test_id, target_column_name, csv_save_path, fold_num,
        cv_mode, metrics_str, model_dict, ensemble_mode, ensemble_model=None,
        dict_model_choose_data=None):
        
    if isinstance(df_labeled, dict):
        df_labeled_data = pd.DataFrame(np.zeros(len(df_target)))   # only used for place holder
    else:
        df_labeled_data = df_labeled

    list_fold_best_metric_score = []
    for fold_index, (train_index, val_index) in enumerate(cv_mode.split(X=df_labeled_data, y=df_target)):
        X_train = df_labeled_data.loc[train_index]
        X_val = df_labeled_data.loc[val_index]
        Y_train = df_target.loc[train_index]
        Y_val = df_target.loc[val_index]

        # train model & metrics score for validation data
        metrics_score_dict = {}
        df_model_pred_train, df_model_pred_val, df_model_pred_test = \
                pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
        for model_name, model in model_dict.items():
            # choose data
            if isinstance(df_labeled, dict):
                df_X_all = df_labeled[dict_model_choose_data[model_name]]
                X_train = df_X_all.loc[train_index]
                X_val = df_X_all.loc[val_index]
                df_test_encoding_data = df_test_encoding[dict_model_choose_data[model_name]]
            else:
                df_test_encoding_data = df_test_encoding
            # train & validation
            print(f"[Debug]: Training for {model_name}...")
            model.fit(X_train, Y_train)
            print(f"[Debug]: Validation for {model_name}...")
            pred_val = model.predict_proba(X_val)[:, 1]
            pred_test = model.predict_proba(df_test_encoding_data)[:, 1]
            val_metric = mlu.METRICS_DICT[metrics_str](Y_val, pred_val)
            metrics_score_dict[model_name] = val_metric
            df_model_pred_val[model_name] = pred_val
            df_model_pred_test[model_name] = pred_test

            # model based ensemble will use predictions on train_data 
            #     to train ensemble model
            if ensemble_model is not None:
                pred_train = model.predict_proba(X_train)[:, 1]
                df_model_pred_train[model_name] = pred_train
        
        # debug string for metrics score
        print(f"[Debug]: {metrics_str} for {fold_index}th fold:")
        for model_name, metrics_score in metrics_score_dict.items():
            print(f"{model_name}: {metrics_score}")

        # ensemble
        print("[Debug]: Ensemble...")
        ensemble_processor = ensemble.EnsembleProcessor(ensemble_mode, metrics_str, 
                ensemble_model, metrics_score_dict)
        df_test_ensemble, best_metric_score = ensemble_processor.find(
                df_model_pred_train, Y_train, df_model_pred_val, Y_val, df_model_pred_test)

        list_fold_best_metric_score.append(best_metric_score)

        # saving
        print("[Debug]: Saving...")
        utils.save_kaggle_csv(
            df_test_id, 
            target_column_name, 
            df_test_ensemble.values, 
            os.path.join(csv_save_path, 
                    f"{fold_num}_{round(best_metric_score, 4)}_{fold_index}.csv"))
    
    print(f"[Debug]: overall cv average {metrics_str}: {np.mean(list_fold_best_metric_score)}")