from sklearn import preprocessing, model_selection

FOLD_NUM = 5
RARE_VALUE_COUNTS = 2000
RARE_VALUE_RATIO = 0.03

encode_dict = {
    "OneHot": preprocessing.OneHotEncoder(),
    "LabelEncode": preprocessing.LabelEncoder()
}

cross_validation_dict = {
    "KFold": model_selection.KFold(n_splits=FOLD_NUM),
    "StratifiedKFold": model_selection.StratifiedKFold(n_splits=FOLD_NUM)
}