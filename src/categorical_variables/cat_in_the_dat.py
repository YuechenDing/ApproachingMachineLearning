import pandas as pd
import argparse
from sklearn import preprocessing, model_selection, linear_model, metrics
import config

def explore_data(dataframe):
    """Use feature dataframe, print feature's infomation"""
    for column in dataframe.columns:
        print("**Length**: {}".format(len(dataframe)))
        print("**Feature category counts**: \n{}".format(dataframe[column].value_counts()))

def preprocess(dataframe_all, fill_na=True, select_fill_na_feature=None, 
        construct_rare_class=False, select_rare_feature=None, categorical_encode=None,
        select_categorical_encode=None):
    if fill_na == True:
        # fill NaN of all the features if select_fill_na_feature is None
        if (select_fill_na_feature is None) or (type(select_fill_na_feature) is list):
            columns = dataframe_all.columns if select_fill_na_feature is None else select_fill_na_feature
            for col in columns:
                dataframe_all.loc[:, col] = dataframe_all.loc[:, col].fillna("None").astype(str)
        else:
            print("Error: select_fill_na_feature should be None or list type")
            return -1
    if construct_rare_class == True:
        # construct rare class for all the features if select_rare_feature is None
        # rare class conditions: class counts < RARE_VALUE_COUNTS || class counts < RARE_VALUE_RATIO * (data_counts // class_num)
        if (select_rare_feature is None) or (type(select_rare_feature) is list):
            columns = dataframe_all.columns if select_rare_feature is None else select_rare_feature
            for col in columns:
                df_counts = dataframe_all[col].value_counts()
                dataframe_all.loc[
                    (df_counts[dataframe_all[col]].values < min(config.RARE_VALUE_COUNTS, 
                            (len(dataframe_all) // len(df_counts)) * config.RARE_VALUE_RATIO)),
                    col
                ] = "Rare"
        else:
            print("Error: select_rare_feature should be None or list type")
            return -1
    if categorical_encode is not None:
        # encode for all features if select_categorical_encode is None
        if (select_categorical_encode is None) or (type(select_categorical_encode) is list):
            columns = dataframe_all.columns if select_categorical_encode is None else select_categorical_encode
            # check categorical_encode
            if categorical_encode not in config.encode_dict:
                print("Error: {} is not in config.encode_dict".format(categorical_encode))
                return -1

            # encoder fit_transform
            encoder = config.encode_dict[categorical_encode]
            if categorical_encode == "OneHot":
                ###### TODO: OneHotEncoder for selected features
                encoder.fit(dataframe_all)
            else:
                for col in columns:
                    dataframe_all.loc[:, col] = encoder.fit_transform(dataframe_all.loc[:, col])

    return dataframe_all, encoder

if __name__ == "__main__":
    # explore data
    print("Reading csv files...")
    df_labeled = pd.read_csv("../../input/cat_in_the_dat/train.csv")
    df_test = pd.read_csv("../../input/cat_in_the_dat/test.csv")
    list_features = [feature for feature in df_labeled.columns if feature not in ["id", "target"]]
    # explore_data(dataframe=df_labeled[list_features])

    # preprocess
    print("Preprocessing...")
    df_all = pd.concat([df_labeled[list_features], df_test[list_features]], axis=0)
    df_all, encoder = preprocess(df_all, fill_na=True, construct_rare_class=False, categorical_encode="OneHot")
    df_target = df_labeled.target
    df_labeled = df_all.head(len(df_labeled))
    df_test = df_all.tail(len(df_test))

    # cross-validation
    stratified_kfold = config.cross_validation_dict["StratifiedKFold"]  ##
    for fold_index, (train_index, val_index) in enumerate(stratified_kfold.split(X=df_labeled, y=df_target)):
        print("Fold: {}".format(fold_index))

        X_train = encoder.transform(df_labeled.loc[train_index, :])
        Y_train = df_target.loc[train_index]
        X_val = encoder.transform(df_labeled.loc[val_index, :])
        Y_val = df_target.loc[val_index]

        print("Training...")
        model = linear_model.LogisticRegression()
        model.fit(X_train, Y_train)

        print("Validation...")
        val_probability = model.predict_proba(X_val)[:, 1]

        auc = metrics.roc_auc_score(Y_val, val_probability)
        print("AUC: {}".format(auc))

