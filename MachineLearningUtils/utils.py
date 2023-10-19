from ast import arg
import pandas as pd
def transform_single_value_to_list(argument):
    if isinstance(argument, (str, int, float)):
        return [argument]
    elif isinstance(argument, list):
        return argument
    else:
        print("Warning: [transform_single_value_to_list] argument: "
                "argument must be single value type or list")
        return -1

def transform_to_value_list(argument, length):
    if check_single_value(argument):
        return [argument] * length
    elif isinstance(argument, list):
        return argument
    else:
        print("Warning: [transform_to_value_list] argument: "
                "argument must be single value type or list")
        return -1

def check_single_value(argument):
    return isinstance(argument, (str, int, float))


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
