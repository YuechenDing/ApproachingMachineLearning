from sklearn import preprocessing, model_selection, ensemble, tree, linear_model
import xgboost as xgb

FOLD_NUM = 600
RARE_VALUE_COUNTS = 2000
RARE_VALUE_RATIO = 0.03
TRAIN_CSV_PATH = "../../input/cat_in_the_dat/train.csv"
TEST_CSV_PATH = "../../input/cat_in_the_dat/test.csv"
OUTPUT_CSV_ROOT_PATH = "../../output/logistic_regression"

encode_dict = {
    "OneHot": preprocessing.OneHotEncoder(),
    "LabelEncode": preprocessing.LabelEncoder()
}

cross_validation_dict = {
    "KFold": model_selection.KFold(n_splits=FOLD_NUM),
    "StratifiedKFold": model_selection.StratifiedKFold(n_splits=FOLD_NUM)
}

model_dict = {
    "LogisticRegression": linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, max_iter=100),
    "DecisionTreeClassifier": tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, min_samples_split=2),
    "RandomForestClassifier": ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, n_jobs=-1),
    "XGBClassifier": xgb.XGBClassifier(max_depth=7, n_estimators=200, n_jobs=-1)
}