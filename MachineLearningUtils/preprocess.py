from MachineLearningUtils import RARE_VALUE_COUNTS,RARE_VALUE_RATIO,ENCODE_DICT,TRANSFORM_DICT
import pandas as pd


def preprocess(dataframe_all, fill_na=True, select_fill_na_feature=None, construct_rare_class=False,
        select_rare_feature=None, categorical_encode=None, select_categorical_encode=None,
        continuous_transform=None, select_contious_transform=None, rare_value_counts=RARE_VALUE_COUNTS,
        rare_value_ratio=RARE_VALUE_RATIO, config_encode_dict=ENCODE_DICT, 
        config_transform_dict=TRANSFORM_DICT):
    if fill_na == True:
        # fill NaN of all the features if select_fill_na_feature is None
        if (select_fill_na_feature is None) or isinstance(select_fill_na_feature, list):
            columns = dataframe_all.columns if select_fill_na_feature is None else select_fill_na_feature
            for col in columns:
                dataframe_all.loc[:, col] = dataframe_all.loc[:, col].astype(str).fillna("None")
        else:
            print("Error: select_fill_na_feature should be None or list type")
            return -1
    if construct_rare_class == True:
        # construct rare class for all the features if select_rare_feature is None
        # rare class conditions: 
        # class counts < RARE_VALUE_COUNTS || class counts < RARE_VALUE_RATIO * (data_counts // class_num)
        if (select_rare_feature is None) or isinstance(select_rare_feature, list):
            columns = dataframe_all.columns if select_rare_feature is None else select_rare_feature
            for col in columns:
                dataframe_all[col] = dataframe_all[col].astype(str)
                df_counts = dataframe_all[col].value_counts()
                dataframe_all.loc[
                    (df_counts[dataframe_all[col]].values < min(rare_value_counts, 
                            (len(dataframe_all) // len(df_counts)) * rare_value_ratio)),
                    col
                ] = "Rare"
        else:
            print("Error: select_rare_feature should be None or list type")
            return -1
    if categorical_encode is not None:
        # check categorical_encode
        if categorical_encode not in config_encode_dict:
            print("Error: {} is not in config_encode_dict".format(categorical_encode))
            return -1
        
        # encode for all features if select_categorical_encode is None
        categorical_encoder = config_encode_dict[categorical_encode]
        if select_categorical_encode is None:
            columns = dataframe_all.columns
            if categorical_encode == "OneHot":
                categorical_encoder.fit(dataframe_all)
            else:
                for col in columns:
                    dataframe_all.loc[:, col] = categorical_encoder.fit_transform(dataframe_all.loc[:, col])
        elif isinstance(select_categorical_encode, (list, str, pd.core.indexes.base.Index)):
            columns = pd.Index([select_categorical_encode]) if isinstance(select_categorical_encode, str) \
                    else pd.Index(select_categorical_encode)
            if categorical_encode == "OneHot":
                categorical_encoder.sparse_output = False
                encodings = categorical_encoder.fit_transform(dataframe_all[columns])
                df_new = pd.DataFrame(encodings, columns=categorical_encoder.get_feature_names_out())
                dataframe_all = pd.concat([dataframe_all, df_new], axis=1)
                dataframe_all = dataframe_all.drop(columns=columns)
            else:
                for col in columns:
                    dataframe_all.loc[:, col] = categorical_encoder.fit_transform(dataframe_all.loc[:, col])
        else:
            print("Error: select_categorical_encode should be None or str or list type")
            return -1
    if continuous_transform is not None:
        # check continuous_transform
        if continuous_transform not in config_transform_dict:
            print("Error: {} is not in config_transform_dict".format(continuous_transform))
            return -1

        # continuous feature transform
        continuous_transform_processer = config_transform_dict[continuous_transform]
        columns = dataframe_all.columns if select_contious_transform is None else select_contious_transform
        if isinstance(columns, (list, str, pd.core.indexes.base.Index)):
            columns = pd.Index([columns]) if isinstance(columns, str) else pd.Index(columns)
            dataframe_all[columns] = continuous_transform_processer.fit_transform(dataframe_all[columns])
        else:
            print("Error: select_categorical_encode should be None or str or list type")
            return -1

    return dataframe_all, categorical_encoder