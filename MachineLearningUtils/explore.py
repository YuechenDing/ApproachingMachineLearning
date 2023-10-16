import pandas as pd

from MachineLearningUtils import IS_CONTINUOUS_RATIO

def feature_explore(dataframe):
    print("**Shape** {}\n**NaN Conditions**\n{}".format(
            dataframe.shape, 
            dataframe.isnull().any(axis=0)))
    continuous_feature_list, categorical_feature_list = [], []

    for column in dataframe.columns:
        values_count = dataframe[column].value_counts()
        if len(values_count) < int(IS_CONTINUOUS_RATIO * len(dataframe)): # categorical
            categorical_feature_list.append(column)
            print("**Feature Name: {}, Unique Value Counts: {}**\n{}"\
                    .format(column, len(values_count), values_count))
        else:
            continuous_feature_list.append(column)
    for column in continuous_feature_list: # continuous
        print(dataframe[column].describe())
    
    print("**continuous_feature_list: {}, length: {}**\n"\
            "**categorical_feature_list: {}, length: {}**".format(
                continuous_feature_list, len(continuous_feature_list),
                categorical_feature_list, len(categorical_feature_list)
            ))