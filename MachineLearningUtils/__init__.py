from sklearn import preprocessing

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