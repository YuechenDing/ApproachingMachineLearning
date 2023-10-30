from sklearn import metrics, model_selection
import MachineLearningUtils as mlu
from MachineLearningUtils import utils, ensemble, model_bank
import numpy as np
import pandas as pd
import torch.nn as nn
import os

class ClassWrapper:
    def __init__(self, model, problem, num_class):
        self.model = model
        self.problem = problem
        self.num_class = num_class
        # check model_type
        if isinstance(model, model_bank.ModelBank):
            self.model_type = "custom"
        elif isinstance(model, nn.Module):
            self.model_type = "torch"
        else:
            self.model_type = "sklearn"
        
    def predict_proba(self, X):
        if self.problem == "classification":
            if self.model_type == "torch" or self.model_type == "custom":
                return self.model(X)
            elif self.model_type == "sklearn" and self.num_class == 2:
                return self.model.predict_proba(X)[:, 1]
            else:
                return self.model.predict_proba(X)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        if self.model_type == "sklearn":
            return self.model.fit(x_train, y_train)
        else:
            return self.model(x_train, y_train, x_val, y_val)

def classification_model_decorator(function):
    def wrapper(*args, **kwargs):
        num_class = len(pd.unique(args[1]))
        kwargs["model"] = ClassWrapper(kwargs["model"], 
                problem="classification",
                num_class=num_class)
        return function(*args, **kwargs)
    return wrapper

@classification_model_decorator
def classification_single_model_cv(df_labeled, df_target, df_test_encoding, cv_mode, *, model, list_metrics,
        df_test_id, target_column_name, csv_save_path, fold_num, validation_save_path=None, log_level=2):
    """
    Utilize all labeled data to train & val & test with cross-validation
    Parameters:
        - list_metrics
            - str or list, calculate all the metrics value in validation
            - first metric is used to generate csv filename
        - df_test_id: used to generate csv
        - target_column_name: Y column name of dataframe
    """
    log_writter = utils.LogWritter(log_level)
    # check value type
    list_metrics = utils.transform_single_value_to_list(list_metrics)
    if isinstance(list_metrics, utils.ErrorStatus):
        log_writter["Error"].print("[classification_single_model_cv]: " + list_metrics.get_message())
        return utils.ErrorStatus("classification_single_model_cv")

    # pipeline
    array_val_metrics = np.zeros((fold_num, len(list_metrics)))  # initialize
    for fold_index, (train_index, val_index) in enumerate(cv_mode.split(X=df_labeled, y=df_target)):
        X_train = df_labeled.loc[train_index]
        X_val = df_labeled.loc[val_index]
        Y_train = df_target.loc[train_index]
        Y_val = df_target.loc[val_index]

        # sklearn-like models use X_train, Y_train for training, set aside part of the data for Validation
        # torch-like models use X_train, Y_train for training, X_val, Y_val for Validation
        log_writter["Debug"].print("Training...")
        model.fit(X_train, Y_train, X_val, Y_val)

        log_writter["Debug"].print("Validation...")
        pred_train = model.predict_proba(X_train)
        pred_val = model.predict_proba(X_val)

        for metric_index, metric in enumerate(list_metrics):
            train_metric = mlu.METRICS_DICT[metric](Y_train, pred_train)
            val_metric = mlu.METRICS_DICT[metric](Y_val, pred_val)
            array_val_metrics[fold_index, metric_index] = val_metric
            log_writter["Debug"].print(f"Train_{metric}: {train_metric}, "
                    f"Val_{metric}: {val_metric}")

        log_writter["Debug"].print("Saving...")
        # save validation predictions
        if validation_save_path != None:
            pd.DataFrame(pred_val).to_csv(
                    os.path.join(validation_save_path, str(fold_index) + ".csv"), 
                    index=False)
        # save test predictions
        utils.save_kaggle_csv(
                df_test_id, 
                target_column_name, 
                model.predict_proba(df_test_encoding), 
                os.path.join(csv_save_path, 
                        f"{fold_num}_{round(array_val_metrics[fold_index, 0], 4)}_{fold_index}.csv"))

    mean_metrics = np.mean(array_val_metrics, axis=0)
    for index, metric in enumerate(list_metrics):
        log_writter["Debug"].print(f"overall cv average {metric}: {mean_metrics[index]}")

@classification_model_decorator
def classification_single_model_full(df_labeled, df_target, df_test_encoding, *, model, list_metrics,
        df_test_id, target_column_name, csv_save_path, log_level=2):
    """
    All the labeled data is used for training
    Parameters:
        - list_metrics
            - str or list, calculate all the metrics value in validation
            - first metric is used to generate csv filename
        - df_test_id: used to generate csv
        - target_column_name: Y column name of dataframe
    """
    log_writter = utils.LogWritter(log_level)
    # check value type
    list_metrics = utils.transform_single_value_to_list(list_metrics)
    if isinstance(list_metrics, utils.ErrorStatus):
        log_writter["Error"].print("[classification_single_model_full] " + list_metrics.get_message())
        return utils.ErrorStatus("classification_single_model_full")

    # pipeline
    array_val_metrics = np.zeros(len(list_metrics))  # initialize

    log_writter["Debug"].print("Training...")
    model.fit(df_labeled, df_target, df_labeled, df_target)

    log_writter["Debug"].print("Training Metrics...")
    pred_train = model.predict_proba(df_labeled)
    for metric_index, metric in enumerate(list_metrics):
        train_metric = mlu.METRICS_DICT[metric](df_target, pred_train)
        array_val_metrics[metric_index] = train_metric
        log_writter["Debug"].print(f"Train_{metric}: {train_metric}")

    log_writter["Debug"].print("Saving...")
    utils.save_kaggle_csv(
            df_test_id, 
            target_column_name, 
            model.predict_proba(df_test_encoding), 
            os.path.join(csv_save_path, 
                    f"{round(array_val_metrics[0], 4)}.csv"))

    for index, metric in enumerate(list_metrics):
        log_writter["Debug"].print(f"overall cv average {metric}: {array_val_metrics[index]}")

def classification_single_model_hyperparam_cv(df_labeled, df_target, df_test_encoding, 
        model, tune_mode, list_metrics, search_param, num_cv,
        df_test_id, target_column_name, csv_save_path, log_level=2):
    log_writter = utils.LogWritter(log_level)
    # check value type
    if not(isinstance(list_metrics, (list, str))):
        log_writter["Error"].print("list_metrics must be list or str type")
        return utils.ErrorStatus("classification_single_model_hyperparam_cv")

    log_writter["Debug"].print("Training...")
    tune_model = mlu.TUNE_MODEL_DICT[tune_mode](
            estimator=model,
            param_grid=search_param,
            scoring=list_metrics,
            verbose=3,
            n_jobs=-1,
            cv=num_cv)
    # tune_model only accepts sklearn-like models
    tune_model.fit(df_labeled, df_target)

    log_writter["Debug"].print("Saving...")
    utils.save_kaggle_csv(
            df_test_id, 
            target_column_name, 
            tune_model.predict_proba(df_test_encoding), 
            os.path.join(csv_save_path, 
                    f"{round(tune_model.best_score_, 4)}.csv"))
                    
    log_writter["Debug"].print("Best params: ")
    best_params = tune_model.best_estimator_.get_params()
    for param_name in search_param.keys():
        log_writter["Debug"].print(f"{param_name}: {best_params[param_name]}")

def classification_ensemble_cv(df_labeled, df_target, df_test_encoding, 
        df_test_id, target_column_name, csv_save_path, fold_num,
        cv_mode, metrics_str, model_dict, ensemble_mode, ensemble_model=None,
        dict_model_choose_data=None, log_level=2):
    log_writter = utils.LogWritter(log_level)
        
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
            model = ClassWrapper(model, "classification", len(pd.unique(df_target)))
            # choose data
            if isinstance(df_labeled, dict):
                df_X_all = df_labeled[dict_model_choose_data[model_name]]
                X_train = df_X_all.loc[train_index]
                X_val = df_X_all.loc[val_index]
                df_test_encoding_data = df_test_encoding[dict_model_choose_data[model_name]]
            else:
                df_test_encoding_data = df_test_encoding
            # train & validation
            log_writter["Debug"].print(f"Training for {model_name}...")
            model.fit(X_train, Y_train, X_val, Y_val)

            log_writter["Debug"].print(f"Validation for {model_name}...")
            pred_val = model.predict_proba(X_val)
            pred_test = model.predict_proba(df_test_encoding_data)
            val_metric = mlu.METRICS_DICT[metrics_str](Y_val, pred_val)
            metrics_score_dict[model_name] = val_metric
            df_model_pred_val[model_name] = pred_val
            df_model_pred_test[model_name] = pred_test

            # model based ensemble will use predictions on train_data 
            #     to train ensemble model
            if ensemble_model is not None:
                pred_train = model.predict_proba(X_train)
                df_model_pred_train[model_name] = pred_train
        
        # debug string for metrics score
        log_writter["Debug"].print(f"{metrics_str} for {fold_index}th fold:")
        for model_name, metrics_score in metrics_score_dict.items():
            log_writter["Debug"].print(f"{model_name}: {metrics_score}")

        # ensemble
        log_writter["Debug"].print("Ensemble...")
        ensemble_processor = ensemble.EnsembleProcessor(ensemble_mode, metrics_str, 
                ensemble_model, metrics_score_dict)
        df_test_ensemble, best_metric_score = ensemble_processor.find(
                df_model_pred_train, Y_train, df_model_pred_val, Y_val, df_model_pred_test)

        list_fold_best_metric_score.append(best_metric_score)

        # saving
        log_writter["Debug"].print("Saving...")
        utils.save_kaggle_csv(
            df_test_id, 
            target_column_name, 
            df_test_ensemble.values, 
            os.path.join(csv_save_path, 
                    f"{fold_num}_{round(best_metric_score, 4)}_{fold_index}.csv"))
    
    log_writter["Debug"].print(f"overall cv average {metrics_str}: {np.mean(list_fold_best_metric_score)}")
