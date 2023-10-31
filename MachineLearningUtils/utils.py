import pandas as pd
from collections.abc import Iterable

from sklearn import utils
import MachineLearningUtils as mlu

class ErrorStatus:
    def __init__(self, function_name, class_name=None, additional_message=None):
        self.class_name = class_name
        self.function_name = function_name
        self.additional_message = additional_message
    def get_message(self):
        result_str = f"{self.function_name} "
        # class_name is optional
        if self.class_name is not None:
            result_str += f" in {self.class_name} class "
        result_str += "failed"
        # additional_message is optional
        if self.additional_message is not None:
            result_str += f", {self.additional_message}"
        return result_str


class LogWritter:
    """
    Print Debug/Warning/Error message.
    print_log_level:
        - 0: Error message only
        - 1: Error + Warning message
        - >=2: Error + Warning + Debug message
    """
    def __init__(self, print_log_level: int = 0):
        self.print_log_level = print_log_level
        self.this_message_log_level = None
        self.this_message_prefix = ""
    def __getitem__(self, message_level_str):
        # check message_level type
        if not (isinstance(message_level_str, str)):
            print("[Error]: [ErrorMessage.__getitem__]: argument: message_level "
                    "must be str type")
            return ErrorStatus("__get_item__", "LogWritter")
        # set this_message_log_level value
        message_level_str = message_level_str.lower()
        self.this_message_log_level = mlu.LOG_LEVEL_DICT[message_level_str] \
                if message_level_str in mlu.LOG_LEVEL_DICT \
                else mlu.LOG_LEVEL_DICT["default"]
        log_rank_str = message_level_str.upper() \
                if message_level_str in mlu.LOG_LEVEL_DICT \
                else "TRACE"
        self.this_message_prefix = "[" + log_rank_str + "]: "
        return self
    def print(self, message_string: str):
        if self.this_message_log_level <= self.print_log_level:
            print(self.this_message_prefix + message_string)
        

def transform_single_value_to_list(argument, log_level=0):
    log_writter = LogWritter(log_level)
    if isinstance(argument, (str, int, float)):
        return [argument]
    elif isinstance(argument, list):
        return argument
    else:
        log_writter["Error"].print("[transform_single_value_to_list] argument: "
                "argument must be single value type or list")
        return ErrorStatus("transform_single_value_to_list")

def transform_to_value_list(argument, length, log_level=0):
    log_writter = LogWritter(log_level)
    if check_single_value(argument):
        return [argument] * length
    elif isinstance(argument, list):
        return argument
    else:
        log_writter["Error"].print("[transform_to_value_list] argument: "
                "argument must be single value type or list")
        return ErrorStatus("transform_to_value_list")

def check_single_value(argument):
    return isinstance(argument, (str, int, float))

def check_transform_single_value_int(argument, log_level=0):
    log_writter = LogWritter(log_level)
    if isinstance(argument, (str, int, float)):
        return int(argument) if isinstance(argument, str) else argument
    else:
        log_writter["Error"].print("[check_transform_single_value]: argument: "
                "argument must be single value type")
        return ErrorStatus("check_transform_single_value_int")

def check_transform_single_value_float(argument, log_level=0):
    log_writter = LogWritter(log_level)
    if isinstance(argument, (str, int, float)):
        return float(argument) if isinstance(argument, str) else argument
    else:
        log_writter["Error"].print("[Error]: [check_transform_single_value]: argument: "
                "argument must be single value type")
        return ErrorStatus("check_transform_single_value_float")

def check_transform_no_nest_iterable_int(argument, log_level=0):
    log_writter = LogWritter(log_level)
    if isinstance(argument, Iterable):
        result = list(map(check_transform_single_value_int, argument))
        if None in result:
            log_writter["Error"].print("Error]: [check_transform_no_nest_iterable]: argument: "
                    "each value in argument must be single value type")
            return ErrorStatus("check_transform_no_nest_iterable_int")
        else:
            return result
    else:
        log_writter["Error"].print("[check_transform_no_nest_iterable]: argument: "
                "argument must be no_nested iterable type")
        return ErrorStatus("check_transform_no_nest_iterable_int")
    
def check_transform_target_tensor(target, input=None, log_level=0):
    """
    transform target to 2-dimensional shape
    input:
        - != None: transform target.shape[0] == input.shape[0]
        - None: target.unsqueeze(1)
    """
    log_writter = LogWritter(log_level)
    if len(target.shape) < 2:
        target = target.unsqueeze(1)
        if input is not None:
            if target.shape[0] != input.shape[0]:
                log_writter["Error"].print("target length should be equal with"
                        " input feature length")
                return ErrorStatus("check_transform_target_tensor")
    return target


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
