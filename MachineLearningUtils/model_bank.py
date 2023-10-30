from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from zmq import device
import MachineLearningUtils as mlu
from MachineLearningUtils import utils
import pandas as pd
from pandas.core.frame import DataFrame, Series

def construct_model_block(in_feature_count, out_feature_count, hidden_layer,
        dropout_position, dropout_probability, activation_layer="ReLU", 
        log_level=0):
    log_writter = utils.LogWritter(log_level)
    ## check value type
    in_feature_count = utils.check_transform_single_value_int(in_feature_count)
    out_feature_count = utils.check_transform_single_value_int(out_feature_count)
    if in_feature_count is None or out_feature_count is None:
        log_writter["Error"].print("[construct_model_block]: check_transform_single_value_int "
                "for argument 'in_feature_count' or 'out_feature_count' failed")
        return utils.ErrorStatus("construct_model_block")
    dropout_probability = utils.check_transform_single_value_float(dropout_probability)
    if dropout_probability is None:
        log_writter["Error"].print("[construct_model_block]: check_transform_single_value_float "
                "for argument 'dropout_probability' failed")
        return utils.ErrorStatus("construct_model_block")
    hidden_layer = utils.check_transform_no_nest_iterable_int(hidden_layer)
    if isinstance(hidden_layer, utils.ErrorStatus):
        log_writter["Error"].print("[construct_model_block]: check_transform_no_nest_iterable_int "
                "for argument 'hidden_layer' failed")
        return utils.ErrorStatus("construct_model_block")
    # dropout_position: None -> None
    #                   no_nest_iterable -> set
    if dropout_position != None:
        dropout_position = utils.check_transform_no_nest_iterable_int(dropout_position)
        if isinstance(dropout_position, utils.ErrorStatus):
            log_writter["Error"].print("[construct_model_block]: check_transform_no_nest_iterable_int "
                    "for argument 'dropout_position' failed")
            return utils.ErrorStatus("construct_model_block")
        dropout_position = set(dropout_position)

    ## construct model block sequence
    layer_feature_count_list = [in_feature_count] + hidden_layer + [out_feature_count]
    in_count_index, out_count_index = 0, 1
    count_layer_block = 0
    result_model = nn.Sequential()
    # construct hidden layer
    while out_count_index < len(layer_feature_count_list) - 1:
        in_count = layer_feature_count_list[in_count_index]
        out_count = layer_feature_count_list[out_count_index]
        # check block
        if (dropout_position is None) or (count_layer_block not in dropout_position):
            temp_block = linear_activation_block(in_count, out_count, activation_layer)
        else:
            ### TODO: set dropout_probability for each position
            temp_block = linear_activation_dropout_block(in_count, out_count, 
                    activation_layer, dropout_probability)

        if isinstance(temp_block, utils.ErrorStatus):
            log_writter["Error"].print("[construct_model_block]: construct linear block failed")
            return utils.ErrorStatus("construct_model_block")

        # extend block
        result_model.extend(temp_block)

        # counter refresh
        in_count_index += 1
        out_count_index += 1
        count_layer_block += 1

    #construct tail layer
    result_model.append(nn.Linear(
            layer_feature_count_list[in_count_index], 
            layer_feature_count_list[out_count_index]))
    
    return result_model

def linear_activation_block(in_feature_count, out_feature_count, activation_layer,
        log_level=0):
    """
        activation_layer: str, mapping fromACTIVATION_DICT
    """
    log_writter = utils.LogWritter(log_level)
    # check activation_layer value
    if activation_layer not in mlu.ACTIVATION_DICT:
        activation_dict_key = ",".join(mlu.ACTIVATION_DICT.keys())
        log_writter["Error"].print(f"[linear_activation_block] activation_layer only supports "
                f"{activation_dict_key}, but '{activation_layer}' is given")
        return utils.ErrorStatus("linear_activation_block")
    
    return nn.Sequential(
            nn.Linear(in_feature_count, out_feature_count, bias=True),
            mlu.ACTIVATION_DICT[activation_layer])

def linear_activation_dropout_block(in_feature_count, out_feature_count,
        activation_layer, dropout_probability, log_level=0):
    log_writter = utils.LogWritter(log_level)
    # check activation_layer value
    if activation_layer not in mlu.ACTIVATION_DICT:
        activation_dict_key = ",".join(mlu.ACTIVATION_DICT.keys())
        log_writter["Error"].print(f"[linear_activation_block] activation_layer only supports "
                f"{activation_dict_key}, but '{activation_layer}' is given")
        return utils.ErrorStatus("linear_activation_dropout_block")

    return nn.Sequential(
            nn.Linear(in_feature_count, out_feature_count),
            nn.Dropout(p=dropout_probability),
            mlu.ACTIVATION_DICT[activation_layer])

class ModelBank(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        super().forward(x)
    def fit(self, x_train, y_train, x_val, y_val):
        pass

class FullyConnected(ModelBank):
    """
    Construct linear block sequence
    Model output: logits (without sigmoid/softmax transform)
    """
    def __init__(self, in_feature_count, out_feature_count, hidden_layer=(100, ), 
            dropout_position=None, dropout_probability=0.1, activation_layer="ReLU",
            epoch=10, batch_size=100, shuffle=True, device='cuda', loss="BCELogits", 
            optimizer="SGD", optimizer_param_dict={},log_level=2, show_log_batch=10):
        super().__init__()
        # check error
        model_block = construct_model_block(in_feature_count, out_feature_count, 
                hidden_layer, dropout_position, dropout_probability, activation_layer)
        if isinstance(model_block, utils.ErrorStatus):
            print("FullyConnected.__init__" + model_block.get_message())
            return utils.ErrorStatus("__init__", "FullyConnected")
        # no error
        self.model_block = model_block
        self.epoch = epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.log_writter = utils.LogWritter(log_level)
        self.show_log_batch = show_log_batch

        if loss not in mlu.LOSS_DICT:
            self.log_writter["Error"].print("[FullyConnected.__init__]: argument: "
                    "loss not in mlu.LOSS_DICT")
            return utils.ErrorStatus("__init__", "FullyConnected")
        self.loss_str = loss
        self.loss = mlu.LOSS_DICT[loss]
        self.optimizer_param_dict = optimizer_param_dict
        if optimizer not in mlu.OPTIMIZER_DICT:
            self.og_writter["Error"].print("[FullyConnected.__init__]: argument: "
                    "optimizer not in mlu.LOSS_DICT")
            return utils.ErrorStatus("__init__", "FullyConnected")
        self.optimizer = optimizer(self.model_block.parameters(), **self.optimizer_param_dict)
        
    def forward(self, x):
        return self.model_block(x)
    
    def predict_proba(self, x):
        return self.model_block(x)

    def fit(self, x_train, y_train, x_val, y_val):
        # check value type
        if isinstance(x_train, DataFrame):
            x_train = x_train.values
        if isinstance(y_train, DataFrame):
            y_train = y_train.values
        if isinstance(x_val, DataFrame):
            x_val = x_val.values
        if isinstance(y_val, DataFrame):
            y_val = y_val.values

        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        x_val = torch.Tensor(x_val)
        y_val = torch.Tensor(y_val)

        # Dataset & DataLoader
        dataset_train = TensorDataset(x_train, y_train)
        dataset_val = TensorDataset(y_train, y_val)
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size,
                shuffle=self.shuffle)
        test_loader = DataLoader(dataset_val, batch_size=self.batch_size)

        # Train & Val
        self.model_block = self.model_block.add_module(self.device)
        for epoch in range(self.epoch):
            # train
            loss_train = torch.Tensor(0.0)
            self.model_block.train()
            for batch_index, (X, Y) in enumerate(train_loader):
                X = X.to(self.device)
                Y = Y.to(self.device)
                self.model_block.train()

                pred = self.forward(X)
                loss = self.loss(pred, Y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    loss_train += loss
                    if batch_index % self.show_log_batch == 0:
                        self.log_writter["Debug"].print(f"Epoch: {epoch}, Batch: {batch_index}/{len(train_loader)}, "
                                f"{self.loss_str}: {loss.item()}")
            self.log_writter["Debug"].print(f"Training {self.loss_str}: {(loss_train/len(train_loader)).item()}")
            
            # val
            self.model_block.eval()
            loss_eval = torch.Tensor(0.0)
            with torch.no_grad():
                for batch_index, (X, Y) in enumerate(test_loader):
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    self.model_block.eval()

                    pred = self.forward(X)
                    loss = self.loss(pred, Y)
                    loss_eval += loss
            self.log_writter["Debug"].print(f"Validation {self.loss_str}: {(loss_eval/len(test_loader)).item()}")