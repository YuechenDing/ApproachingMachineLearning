from turtle import forward
import torch
import torch.nn as nn
import MachineLearningUtils as mlu
from MachineLearningUtils import utils

def construct_model_block(in_feature_count, out_feature_count, hidden_layer,
        dropout_position, dropout_probability, activation_layer="ReLU"):
    ## check value type
    in_feature_count = utils.check_transform_single_value_int(in_feature_count)
    out_feature_count = utils.check_transform_single_value_int(out_feature_count)
    if in_feature_count is None or out_feature_count is None:
        print("[Error]: [construct_model_block]: check_transform_single_value_int "
                "for argument 'in_feature_count' or 'out_feature_count' failed")
        return -1
    dropout_probability = utils.check_transform_single_value_float(dropout_probability)
    if dropout_probability is None:
        print("[Error]: [construct_model_block]: check_transform_single_value_float "
                "for argument 'dropout_probability' failed")
        return -1
    hidden_layer = utils.check_transform_no_nest_iterable_int(hidden_layer)
    if hidden_layer == -1:
        print("[Error]: [construct_model_block]: check_transform_no_nest_iterable_int "
                "for argument 'hidden_layer' failed")
        return -1
    # dropout_position: None -> None
    #                   no_nest_iterable -> set
    if dropout_position != None:
        dropout_position = utils.check_transform_no_nest_iterable_int(dropout_position)
        if dropout_position == -1:
            print("[Error]: [construct_model_block]: check_transform_no_nest_iterable_int "
                    "for argument 'dropout_position' failed")
            return -1
        dropout_position = set(dropout_position)

    ## construct model block sequence
    layer_feature_count_list = [in_feature_count] + hidden_layer + [out_feature_count]
    in_count_index, out_count_index = 0, 1
    count_layer_block = 0
    result_model = nn.Sequential()
    while out_count_index < len(layer_feature_count_list):
        in_count = layer_feature_count_list[in_count_index]
        out_count = layer_feature_count_list[out_count_index]
        # check block
        if (dropout_position is None) or (count_layer_block not in dropout_position):
            temp_block = linear_activation_block(in_count, out_count, activation_layer)
        else:
            ### TODO: set dropout_probability for each position
            temp_block = linear_activation_dropout_block(in_count, out_count, 
                    activation_layer, dropout_probability)
        if temp_block == -1:
            print("[Error]: [construct_model_block]: construct linear block failed")
            return -1

        # extend block
        result_model.extend(temp_block)

        # counter refresh
        in_count_index += 1
        out_count_index += 1
        count_layer_block += 1

        print(result_model)

    return result_model

def linear_activation_block(in_feature_count, out_feature_count, activation_layer):
    """
        activation_layer: str, mapping fromACTIVATION_DICT
    """
    # check activation_layer value
    if activation_layer not in mlu.ACTIVATION_DICT:
        activation_dict_key = ",".join(mlu.ACTIVATION_DICT.keys())
        print(f"[Error]: [linear_activation_block] activation_layer only supports "
                f"{activation_dict_key}, but '{activation_layer}' is given")
        return -1
    
    return nn.Sequential(
            nn.Linear(in_feature_count, out_feature_count, bias=True),
            mlu.ACTIVATION_DICT[activation_layer])
def linear_activation_dropout_block(in_feature_count, out_feature_count,
        activation_layer, dropout_probability):
    # check activation_layer value
    if activation_layer not in mlu.ACTIVATION_DICT:
        activation_dict_key = ",".join(mlu.ACTIVATION_DICT.keys())
        print(f"[Error]: [linear_activation_block] activation_layer only supports "
                f"{activation_dict_key}, but '{activation_layer}' is given")
        return -1

    return nn.Sequential(
            nn.Linear(in_feature_count, out_feature_count),
            nn.Dropout(p=dropout_probability),
            mlu.ACTIVATION_DICT[activation_layer])

class FullyConnected(nn.Module):
    """
    Construct linear block sequence
    Model output: logits (without sigmoid/softmax transform)
    """
    def __init__(self, in_feature_count, out_feature_count, hidden_layer=(100, ), 
            dropout_position=None, dropout_probability=0.1, activation_layer="ReLU"):
        super().__init__()
        self.model_block = construct_model_block(in_feature_count, out_feature_count, 
                hidden_layer, dropout_position, dropout_probability, activation_layer)
        
    def forward(self, x):
        return self.model_block(x)
