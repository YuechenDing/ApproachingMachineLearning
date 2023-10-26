import torch
import torch.nn as nn
import MachineLearningUtils as mlu


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
    def __init__(self, hidden_layer=(100, ), dropout_position=None, 
            dropout_probability=0.1, problem="BinaryClassification"):
        super().__init__()
        