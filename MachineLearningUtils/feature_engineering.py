from MachineLearningUtils import utils
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def feature_overview(dataframe, corr_method="pearson", show_fig=True):
    """
    only for continuous features
    calculate variance() corr() of dataframe
    """
    print(dataframe.var())
    df_corr = dataframe.corr(method=corr_method, numeric_only=True)
    if show_fig == False:
        print(df_corr)
    else:
        sns.heatmap(df_corr,
                annot=True,  # display annotations
                center=0.5,
                fmt='.2f',
                linewidth=0.5,  # box width
                linecolor='black',
                vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True,
                square=True,
                cbar=True,  # color bar for corr value
                cmap='coolwarm_r')
        plt.show()

def clear_unselected_columns(dataframe, catecorical_list=None, drop_categorical=None,
        continuous_list=None, drop_continuous=None):
    list_all_drops = []
    # drop catecorical_list
    if isinstance(catecorical_list, list):
        list_all_drops.extend(drop_categorical)
        if isinstance(drop_categorical, list):
            for column in drop_categorical:
                catecorical_list.remove(column)
    # drop continuous_list
    if isinstance(continuous_list, list):
        list_all_drops.extend(drop_continuous)
        if isinstance(drop_continuous, list):
            for column in drop_continuous:
                continuous_list.remove(column)
    # drop dataframe
    dataframe = dataframe.drop(columns=list_all_drops)
    return dataframe

def log_transform(dataframe, columns, replace=True, continuous_list=None):
    # check type
    columns = utils.transform_single_value_to_list(columns)
    if isinstance(columns, utils.ErrorStatus):
        print(columns.get_message())
        return utils.ErrorStatus("log_transform")
    
    if replace == True:
        dataframe[columns] = dataframe[columns].apply(lambda x: np.log(1 + x))
    else:
        new_columns = list(map(lambda x: x + "_log", columns))
        dataframe[new_columns] = dataframe[columns].apply(lambda x: np.log(1 + x))
        continuous_list.extend(new_columns)
    
    return dataframe

def bin_transform(dataframe, columns, bin_amount, replace=False, 
        continuous_list=None, categorical_list=None, log_level=1):
    log_writter = utils.LogWritter(log_level)
    # check columns value type
    # check_transform bin_amount value type
    columns = utils.transform_single_value_to_list(columns)
    bin_amount = utils.transform_to_value_list(bin_amount, len(columns))
    if isinstance(columns, utils.ErrorStatus) or isinstance(bin_amount, utils.ErrorStatus):
        log_writter["Error"].print("[bin_transform] argument: columns, bin_amount "
                "must be list or single value type")
        return utils.ErrorStatus("bin_transform")
    # check length
    if len(columns) != len(bin_amount):
        log_writter["Error"].print("[bin_transform] argument: columns and bin_amount "
                "must have same length")
        return utils.ErrorStatus("bin_transform")
    
    if replace == True:
        ## TODO: bin_transform with relace
        log_writter["Warning"].print("[bin_transform] replace == True not supported")
        pass
    else:
        for col_index, col in enumerate(columns):
            new_column_name = col + "_bin"
            dataframe[new_column_name] = pd.cut(dataframe[col], bin_amount[col_index], labels=False)
            categorical_list.append(new_column_name)
            
        return dataframe