from sklearn import preprocessing, metrics, model_selection

IS_CONTINUOUS_RATIO = 0.9
RARE_VALUE_COUNTS = 2000
RARE_VALUE_RATIO = 0.03

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