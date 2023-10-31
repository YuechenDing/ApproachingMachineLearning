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
from pandas.core.frame import DataFrame
from pandas.core.series import Series

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

def Dataframe2Array(function):
    def wrapper(*args, **kwargs):
        transform_args = []
        for arg in args:
            if isinstance(arg, (DataFrame, Series)):
                arg = arg.values
            transform_args.append(arg)
        return function(*transform_args, **kwargs)
    return wrapper
def DataFrame2Tensor(function):
    def wrapper(*args, **kwargs):
        transform_args = []
        for arg in args:
            if isinstance(arg, (DataFrame, Series)):
                arg = torch.Tensor(arg.values)
            transform_args.append(arg)
        return function(*transform_args, **kwargs)
    return wrapper

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
            sklearn_loss="ROC_AUC", optimizer="SGD", optimizer_param_dict={},
            lr_scheduler="ExponentialLR", lr_scheduler_param_dict={},
            log_level=2, show_log_batch=10, is_classification=True):
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
        self.is_classification = is_classification

        if loss not in mlu.LOSS_DICT:
            self.log_writter["Error"].print("[FullyConnected.__init__]: argument: "
                    "loss not in mlu.LOSS_DICT")
            return utils.ErrorStatus("__init__", "FullyConnected")
        self.loss_str = loss
        self.loss = mlu.LOSS_DICT[loss]
        self.sklearn_loss_str = sklearn_loss
        if self.sklearn_loss_str is not None:
            if self.sklearn_loss_str not in mlu.LOSS_DICT:
                self.log_writter["Error"].print("[FullyConnected.__init__]: argument: "
                    "sklearn_loss not in mlu.LOSS_DICT")
                return utils.ErrorStatus("__init__", "FullyConnected")
            self.sklearn_loss = mlu.LOSS_DICT[sklearn_loss]

        self.optimizer_param_dict = optimizer_param_dict
        if optimizer not in mlu.OPTIMIZER_DICT:
            self.log_writter["Error"].print("[FullyConnected.__init__]: argument: "
                    "optimizer not in mlu.LOSS_DICT")
            return utils.ErrorStatus("__init__", "FullyConnected")
        self.optimizer = mlu.OPTIMIZER_DICT[optimizer](self.model_block.parameters(), **self.optimizer_param_dict)

        self.scheduler_str = lr_scheduler
        if self.scheduler_str is not None:
            if self.scheduler_str not in mlu.SCHEDULER_DICT:
                self.log_writter["Error"].print("[FullyConnected.__init__]: argument: "
                        "lr_scheduler not in mlu.SCHEDULER_DICT")
                return utils.ErrorStatus("__init__", "FullyConnected")
            self.scheduler = mlu.SCHEDULER_DICT[self.scheduler_str](self.optimizer, **lr_scheduler_param_dict)
    
    @DataFrame2Tensor
    def forward(self, x):
        return self.model_block(x)
    
    @DataFrame2Tensor
    def predict_proba(self, x):
        with torch.no_grad():
            self.model_block = self.model_block.to(self.device)
            self.model_block.eval()
            x = x.to(self.device)
            return self.model_block(x).cpu().numpy()

    @Dataframe2Array
    def fit(self, x_train, y_train, x_val, y_val):
        x_train = torch.Tensor(x_train)
        x_val = torch.Tensor(x_val)
        y_train = utils.check_transform_target_tensor(
                torch.Tensor(y_train), x_train)
        if isinstance(y_train, utils.ErrorStatus):
            self.log_writter["Error"].print(
                    "[FullyConnected.fit]: y_train: " + y_train.get_message())
            return utils.ErrorStatus("fit", "FullyConnected")
        y_val = utils.check_transform_target_tensor(
                torch.Tensor(y_val), x_val)
        if isinstance(y_val, utils.ErrorStatus):
            self.log_writter["Error"].print(
                    "[FullyConnected.fit]: y_val: " + y_val.get_message())
            return utils.ErrorStatus("fit", "FullyConnected")

        # Dataset & DataLoader
        dataset_train = TensorDataset(x_train, y_train)
        dataset_val = TensorDataset(x_val, y_val)
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size,
                shuffle=self.shuffle)
        test_loader = DataLoader(dataset_val, batch_size=self.batch_size)

        # Train & Val
        self.model_block = self.model_block.to(self.device)
        for epoch in range(self.epoch):
            # train
            loss_train = 0.0
            self.model_block.train()
            for batch_index, (X, Y) in enumerate(train_loader):
                X = X.to(self.device)
                Y = Y.to(self.device)
                self.model_block.train()
                self.optimizer.zero_grad()

                pred = self.forward(X)
                loss = self.loss(pred, Y)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    loss_train += loss.item()
                    if (self.show_log_batch is not None) \
                            and (batch_index % self.show_log_batch == 0):
                        self.log_writter["Debug"].print(f"Epoch: {epoch}, Batch: {batch_index}/{len(train_loader)}, {self.loss_str}: {loss.item()}")
            self.log_writter["Debug"].print(f"Epoch: {epoch}, Training {self.loss_str}: {loss_train/len(train_loader)}")
            
            # lr scheduler
            if self.scheduler_str is not None:
                self.scheduler.step()
            
            # val
            self.model_block.eval()
            with torch.no_grad():
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                self.model_block.eval()

                pred = self.forward(x_val)
                self.log_writter["Debug"].print(f"Epoch: {epoch}, Validation {self.loss_str}: {(self.loss(pred, y_val)).item()}")
                Y = y_val.cpu().int().numpy() if self.is_classification else y_val.cpu().numpy()
                self.log_writter["Debug"].print(f"Epoch: {epoch}, Validation {self.sklearn_loss_str}: {self.sklearn_loss(Y, pred.cpu().numpy())}")