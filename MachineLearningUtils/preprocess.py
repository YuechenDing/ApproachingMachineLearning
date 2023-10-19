from MachineLearningUtils import RARE_VALUE_COUNTS,RARE_VALUE_RATIO,ENCODE_DICT,TRANSFORM_DICT
import pandas as pd

def fill_na_category(dataframe_all, fill_na_mode="constant", fill_constance="None",
        select_fill_na_categorical_feature=None):
    """
    fill NaN of categorical features
    fill_na_mode:
        - "constant": fill NaN by given fill_constance, transform to str type
    select_fill_na_categorical_feature:
        - None: all the columns will be filled
        - list type: selected columns will be filled
    """
    # check select_fill_na_categorical_feature
    if select_fill_na_categorical_feature is None:
        select_fill_na_categorical_feature = dataframe_all.columns
    elif not isinstance(select_fill_na_categorical_feature, list):
        print("Error: [fill_na_category] argument: select_fill_na_categorical_feature should be None or list")
        return -1

    if fill_na_mode == "constant":
        dataframe_all[select_fill_na_categorical_feature] = \
                dataframe_all[select_fill_na_categorical_feature].fillna(fill_constance).astype(str)
    else:
        ## TODO: fill NaN by knn or other model
        print("Warning: fill_na_mode only support 'constant' right now")
        pass

    return dataframe_all


def preprocess(dataframe_all, fill_na=False, dict_fill_na_category=None, construct_rare_class=False,
        select_rare_feature=None, categorical_encode=None, select_categorical_encode=None,
        continuous_transform=None, select_contious_transform=None, rare_value_counts=RARE_VALUE_COUNTS,
        rare_value_ratio=RARE_VALUE_RATIO, rare_mode="max_hybrid", config_encode_dict=ENCODE_DICT, 
        config_transform_dict=TRANSFORM_DICT):
    print("[preprocess]: start to fill NaN...")
    if fill_na == True:
        # fill NaN categorical & continuous
        dataframe_all = fill_na_category(dataframe_all, **dict_fill_na_category)

    print("[preprocess]: start to construct rare class...")
    if construct_rare_class == True:
        # construct rare class for all the features if select_rare_feature is None
        # rare class conditions: 
        # class counts < RARE_VALUE_COUNTS || class counts < RARE_VALUE_RATIO * (data_counts // class_num)
        if (select_rare_feature is None) or isinstance(select_rare_feature, list):
            columns = dataframe_all.columns if select_rare_feature is None else select_rare_feature
            for col in columns:
                dataframe_all[col] = dataframe_all[col].astype(str)
                df_counts = dataframe_all[col].value_counts()

                # fill rare value by rare_mode
                rare_mode = rare_mode.lower()
                if rare_mode == "rigid":
                    dataframe_all.loc[
                        (df_counts[dataframe_all[col]].values < rare_value_counts),
                        col
                    ] = "Rare"
                elif rare_mode == "max_hybrid":
                    dataframe_all.loc[
                        (df_counts[dataframe_all[col]].values < max(rare_value_counts, 
                                (len(dataframe_all) // len(df_counts)) * rare_value_ratio)),
                        col
                    ] = "Rare"
                elif rare_mode == "min_hybrid":
                    dataframe_all.loc[
                        (df_counts[dataframe_all[col]].values < min(rare_value_counts, 
                                (len(dataframe_all) // len(df_counts)) * rare_value_ratio)),
                        col
                    ] = "Rare"
                else:
                    print("Error: [preprocess] argument: rare_mode should be "
                            "'rigid', 'max_hybrid', 'min_hybrid'")
                    return -1

                # debug infomation
                print(f"After select rare: {col} -> {len(dataframe_all[col].value_counts())}")

        else:
            print("Error: select_rare_feature should be None or list type")
            return -1
    
    print("[preprocess]: start to encode categorical features...")
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
                    dataframe_all[col] = categorical_encoder.fit_transform(dataframe_all[col])
        else:
            print("Error: select_categorical_encode should be None or str or list type")
            return -1

    print("[preprocess]: start to transform continuous features...")
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