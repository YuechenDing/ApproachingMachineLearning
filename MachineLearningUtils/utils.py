from ast import arg
import pandas as pd
from collections.abc import Iterable

def transform_single_value_to_list(argument):
    if isinstance(argument, (str, int, float)):
        return [argument]
    elif isinstance(argument, list):
        return argument
    else:
        print("[Error]: [transform_single_value_to_list] argument: "
                "argument must be single value type or list")
        return -1

def transform_to_value_list(argument, length):
    if check_single_value(argument):
        return [argument] * length
    elif isinstance(argument, list):
        return argument
    else:
        print("[Error]: [transform_to_value_list] argument: "
                "argument must be single value type or list")
        return -1

def check_single_value(argument):
    return isinstance(argument, (str, int, float))

def check_transform_single_value_int(argument):
    if isinstance(argument, (str, int, float)):
        return int(argument) if isinstance(argument, str) else argument
    else:
        print("[Error]: [check_transform_single_value]: argument: "
                "argument must be single value type")
        return None
def check_transform_single_value_float(argument):
    if isinstance(argument, (str, int, float)):
        return float(argument) if isinstance(argument, str) else argument
    else:
        print("[Error]: [check_transform_single_value]: argument: "
                "argument must be single value type")
        return None

def check_transform_no_nest_iterable_int(argument):
    if isinstance(argument, Iterable):
        result = list(map(check_transform_single_value_int, argument))
        if None in result:
            print("[Error]: [check_transform_no_nest_iterable]: argument: "
                    "each value in argument must be single value type")
            return -1
        else:
            return result
    else:
        print("[Error]: [check_transform_no_nest_iterable]: argument: "
                "argument must be no_nested iterable type")
        return -1
    

def save_kaggle_csv(test_id, Y_column_name, prediction_probability, df_output_path):
    """
    test_id: Series (id of test dataframe)
    Y_column_name: target column name
    prediction_probability: predict_proba of the model
    df_output_path: submission.csv path
    """
    df_result = pd.DataFrame(test_id)
    df_result[Y_column_name] = prediction_probability
    df_result.to_csv(df_output_path, index=False)
