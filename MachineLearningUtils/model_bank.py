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
import copy

def construct_model_block(in_feature_count, out_feature_count, hidden_layer,
        dropout_position=None, dropout_probability=0.1, activation_layer="ReLU", 
        log_level=0):
    """
    Construct NN model block, constructed by linear/linear_dropout blocks.
    Used in FullyConnected class.
    Arguments:
        - dropout_position: no nested Iterable int type/None, set layer index of
            dropout
        - dropout_probability: float, dropout layer's probability if dropout is used
        - activation_layer: map from mlu.ACTIVATION_DICT
    """
    log_writter = utils.LogWritter(log_level)
    ## check value type
    in_feature_count = utils.check_transform_single_value_int(in_feature_count)
    out_feature_count = utils.check_transform_single_value_int(out_feature_count)
    if isinstance(in_feature_count, utils.ErrorStatus) or \
            isinstance(out_feature_count, utils.ErrorStatus):
        log_writter["Error"].print("[construct_model_block]: check_transform_single_value_int "
                "for argument 'in_feature_count' or 'out_feature_count' failed")
        return utils.ErrorStatus("construct_model_block")
    hidden_layer = utils.check_transform_no_nest_iterable_int(hidden_layer)
    if isinstance(hidden_layer, utils.ErrorStatus):
        log_writter["Error"].print("[construct_model_block]: check_transform_no_nest_iterable_int "
                "for argument 'hidden_layer' failed")
        return utils.ErrorStatus("construct_model_block")
    # dropout_position: None -> None
    #                   no_nest_iterable -> set
    if dropout_position is not None:
        dropout_position = utils.check_transform_no_nest_iterable_int(dropout_position)
        if isinstance(dropout_position, utils.ErrorStatus):
            log_writter["Error"].print("[construct_model_block]: check_transform_no_nest_iterable_int "
                    "for argument 'dropout_position' failed")
            return utils.ErrorStatus("construct_model_block")
        dropout_position = set(dropout_position)

        dropout_probability = utils.check_transform_single_value_float(dropout_probability)
        if isinstance(dropout_probability, utils.ErrorStatus):
            log_writter["Error"].print("[construct_model_block]: check_transform_single_value_float "
                    "for argument 'dropout_probability' failed")
            return utils.ErrorStatus("construct_model_block")

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
    Construct Linear block layer: Linear + Activation
    Arguments:
        activation_layer: str, mapping from mlu.ACTIVATION_DICT
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
    """
    Construct Linear_Dropout block layer: Linear + Dropout + Activation
    Arguments:
        activation_layer: str, mapping from mlu.ACTIVATION_DICT
    """
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
    """
    Customized models can inherit this class, used for checking type
    e.g. 
        model = FullyConnected(...)
        isinstance(model, ModelBank)
    """
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
    Functions:
        -- decorators are used to transform DataFrame,Series to Tensor 
        - forward(x): 
            - used in training
        - predict_proba(x): 
            - used in validation, already torch.no_grad
        - fit(x_train, y_train, x_val, y_val): training process
            - save model with better score on val_data each epoch
            - log_control: train/val/is_better_log
            - lr_scheduler, optimizer
            - loss/metrics: loss for train, loss/sklearn_loss for val
        - reset_construct_model: 
            - used to reload model each time in fit()
            - commonly used in k-fold train/val:
                e.g.:
                for index, (X_index, Y_index) in enumerate(KFold.split(df_train, df_test)):
                    X, Y, X_val, Y_val = ...
                    model.fit(X, Y, X_val, Y_val)
    """
    def __init__(self, in_feature_count, out_feature_count, hidden_layer=(100, ), 
            dropout_position=None, dropout_probability=0.1, activation_layer="ReLU",
            epoch=10, batch_size=100, shuffle=True, device='cuda', loss="BCELogits", 
            sklearn_loss="ROC_AUC", save_metric="sklearn_loss", optimizer="SGD", 
            optimizer_param_dict={}, lr_scheduler="ExponentialLR", lr_scheduler_param_dict={},
            log_level=2, show_batch_log=10, train_log=True, val_better_log=True,
            is_classification=True):
        super().__init__()
        self.in_feature_count = in_feature_count
        self.out_feature_count = out_feature_count
        self.hidden_layer = hidden_layer
        self.dropout_position = dropout_position
        self.dropout_probability = dropout_probability
        self.activation_layer = activation_layer
        self.epoch = epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.log_writter = utils.LogWritter(log_level)
        self.show_batch_log = show_batch_log
        self.is_classification = is_classification
        self.save_metric = save_metric
        self.train_log = train_log
        self.val_better_log = val_better_log

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
        
        self.optimizer_str = optimizer
        self.optimizer_param_dict = optimizer_param_dict
        self.scheduler_str = lr_scheduler
        self.lr_scheduler_param_dict = lr_scheduler_param_dict
        self.reset_construct_model()
        

    def reset_construct_model(self):
        """
        Used with K-fold train/val, reset model for each fold's train/val
        """
        # check error
        model_block = construct_model_block(
                self.in_feature_count, self.out_feature_count, 
                self.hidden_layer, self.dropout_position, self.dropout_probability, 
                self.activation_layer)
        if isinstance(model_block, utils.ErrorStatus):
            self.log_writter["Error"].print("FullyConnected.reset_construct_model" + model_block.get_message())
            return utils.ErrorStatus("reset_construct_model", "FullyConnected")
        # no error
        self.model_block = model_block

        if self.optimizer_str not in mlu.OPTIMIZER_DICT:
            self.log_writter["Error"].print("[FullyConnected.reset_construct_model]: argument: "
                    "optimizer not in mlu.LOSS_DICT")
            return utils.ErrorStatus("reset_construct_model", "FullyConnected")
        self.optimizer = mlu.OPTIMIZER_DICT[self.optimizer_str](self.model_block.parameters(), **self.optimizer_param_dict)

        if self.scheduler_str is not None:
            if self.scheduler_str not in mlu.SCHEDULER_DICT:
                self.log_writter["Error"].print("[FullyConnected.reset_construct_model]: argument: "
                        "lr_scheduler not in mlu.SCHEDULER_DICT")
                return utils.ErrorStatus("reset_construct_model", "FullyConnected")
            self.scheduler = mlu.SCHEDULER_DICT[self.scheduler_str](self.optimizer, **self.lr_scheduler_param_dict)

    
    @utils.DataFrame2Tensor
    def forward(self, x):
        return self.model_block(x)
    
    @utils.DataFrame2Tensor
    def predict_proba(self, x):
        with torch.no_grad():
            self.model_block = self.model_block.to(self.device)
            self.model_block.eval()
            x = x.to(self.device)
            return self.model_block(x).cpu().numpy()

    @utils.Dataframe2Array
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
        train_loader = DataLoader(dataset_train, batch_size=self.batch_size,
                shuffle=self.shuffle)

        # Train & Val
        self.reset_construct_model()
        best_model = copy.deepcopy(self.model_block)
        best_val_metric = 0
        self.model_block = self.model_block.to(self.device)
        for epoch in range(self.epoch):
            # train
            loss_train = 0.0
            self.model_block.train()
            for batch_index, (X, Y) in enumerate(train_loader):
                X = X.to(self.device)
                Y = Y.to(self.device)
                self.optimizer.zero_grad()

                pred = self.model_block.forward(X)
                loss = self.loss(pred, Y)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    loss_train += loss.item()
                    if (self.show_batch_log is not None) \
                            and (batch_index % self.show_batch_log == 0):
                        self.log_writter["Debug"].print(f"Epoch: {epoch}, Batch: {batch_index}/{len(train_loader)}, {self.loss_str}: {loss.item()}")
            
            # traing epoch log
            if self.train_log:
                self.log_writter["Debug"].print(f"Epoch: {epoch}, Training {self.loss_str}: {loss_train/len(train_loader)}")
            
            # lr scheduler
            if self.scheduler_str is not None:
                self.scheduler.step()
            
            # val
            self.model_block.eval()
            with torch.no_grad():
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)

                pred = self.model_block.forward(x_val)
                loss = (self.loss(pred, y_val)).item()
                log_string = f"Epoch: {epoch}, Validation:\n{self.loss_str}: {loss}"
                
                if self.sklearn_loss is not None:
                    Y = y_val.cpu().int().numpy() if self.is_classification else y_val.cpu().numpy()
                    sklearn_loss = self.sklearn_loss(Y, pred.cpu().numpy())
                    log_string += f", {self.sklearn_loss_str}: {sklearn_loss}"
                # save model
                # TODO: general comparing method
                is_better_flag = False
                if self.save_metric == "sklearn_loss" and self.sklearn_loss is not None:
                    if sklearn_loss > best_val_metric:
                        is_better_flag = True
                        self.log_writter["Debug"].print("Find better model...")
                        best_model = copy.deepcopy(self.model_block)
                        best_val_metric = sklearn_loss
                else:
                    if loss < best_val_metric:
                        is_better_flag = True
                        self.log_writter["Debug"].print("Find better model...")
                        best_model = copy.deepcopy(self.model_block)
                        best_val_metric = loss
                if (self.val_better_log == False) or \
                        (self.val_better_log and is_better_flag):
                    self.log_writter["Debug"].print(log_string)
        # save best model
        self.model_block = best_model