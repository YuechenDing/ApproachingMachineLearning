"""
Categorical: 
    - Nominal; Ordinal; Binary; Cyclic
    - Encoding: Label Encoding; Binarized; One-hot; (CSR Sparse Matrix representing)
    - Replacement: Count Feature; Group Feature; Group Count Feature (search for best group)
Data CLeaning:
    - Unknown(NaN): 
        - Categorical: fill NaN with Unknown category; predict unknown using other features(knn, NN, DT, ...)
        - Numerical: predict unknown using other features
        - Use test together with train to decide the unknown; replicate all the preprocesses in cross validation
        - Add Unknown category in live systems(new categories will be treated as Unknown)
    - Rare:
        - a category less than 2k can be treated as Rare
Procedure:
    - Explore the data
    - Preprocess(fill/predict NaN; Categorical Encoding(None, Rare))
    - Dataset(Cross validation)
    - Model; Training; Validation(metrics)
"""

import pandas as pd
from sklearn import preprocessing, model_selection, metrics, linear_model

## explore data
print("Reading csv files...")
df_labeled = pd.read_csv("../../input/cat_in_the_dat/train.csv")
df_test = pd.read_csv("../../input/cat_in_the_dat/test.csv")
print("labeled dataframe head: ")
print(df_labeled.head())
print("count features: ")
list_features = [feature for feature in df_labeled.columns if feature not in ["id", "target"]]
df_test.loc[:, "target"] = -1
df_all = pd.concat(
        [df_labeled, df_test], 
        axis=0
)
for col in list_features:
    print(df_all[col].value_counts())  # all the features are categorical

## preprocess
# fill NaN
for col in list_features:
    df_all.loc[:, col] = df_all.loc[:, col].fillna("None").astype(str)
df_labeled = df_all[df_all.target != -1].reset_index(drop=True)
df_test = df_all[df_all.target == -1].reset_index(drop=True)
# OneHotEncoding
one_hot_encoding = preprocessing.OneHotEncoder()
one_hot_encoding.fit(df_all[list_features])

## cross-validation
stratified_kfold = model_selection.StratifiedKFold(n_splits=5)
for fold_index, (train_index, val_index) in enumerate(stratified_kfold.split(X=df_labeled, y=df_labeled.target)):
    print("Fold: {}".format(fold_index))

    df_train = df_labeled.loc[train_index, :]
    df_val = df_labeled.loc[val_index, :]

    print("Transforming categorical encodings...")
    x_train = one_hot_encoding.transform(df_train[list_features])
    x_val = one_hot_encoding.transform(df_val[list_features])

    print("Training model...")
    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target)

    print("Validation...")
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    val_probability = model.predict_proba(x_val)[:, 1]

    train_log_loss = metrics.log_loss(df_train.target.values, pred_train)
    val_log_loss = metrics.log_loss(df_val.target.values, pred_val)
    f1 = metrics.f1_score(df_val.target.values, pred_val)
    auc = metrics.roc_auc_score(df_val.target.values, val_probability)
    print("Train_log_loss: {}\nVal_log_loss: {}\nVal_f1_score: {}\n" \
            "Val_AUC: {}".format(train_log_loss, val_log_loss, f1, auc))