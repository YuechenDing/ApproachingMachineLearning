from sklearn import preprocessing, metrics, model_selection
import torch.nn as nn
from torch import optim

IS_CONTINUOUS_RATIO = 0.9
RARE_VALUE_COUNTS = 2000
RARE_VALUE_RATIO = 0.03

LOG_LEVEL_DICT = {
    "default": 0,
    "debug": 2,
    "warning": 1,
    "error": 0
}

ENCODE_DICT = {
    "OneHot": preprocessing.OneHotEncoder(),
    "LabelEncode": preprocessing.LabelEncoder()
}

TRANSFORM_DICT = {
    "MinMax": preprocessing.MinMaxScaler()
}

METRICS_DICT = {
    "roc_auc": metrics.roc_auc_score,
    "accuracy": metrics.accuracy_score
}

TUNE_MODEL_DICT = {
    "grid_search": model_selection.GridSearchCV
}

ENSEMBLE_MODE_SET = {"model_ensemble", "climb_hill"}

ACTIVATION_DICT = {
    "ELU": nn.ELU(),
    "GLU": nn.GLU(),
    "Mish": nn.Mish(),
    "ReLU": nn.ReLU()
}

LOSS_DICT = {
    "BCE": nn.BCELoss(),
    "BCELogits": nn.BCEWithLogitsLoss(),
    "SoftMargin": nn.SoftMarginLoss(),
    "ROC_AUC": metrics.roc_auc_score
}

OPTIMIZER_DICT = {
    "SGD": optim.SGD,
    "Adam": optim.Adam
}

SCHEDULER_DICT = {
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts
}