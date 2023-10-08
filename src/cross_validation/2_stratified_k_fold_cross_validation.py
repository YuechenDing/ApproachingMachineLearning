"""
StratifiedKFold (classification, split)
KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
StratifiedKFold for regression: pd.cut into several bins(classes) 

"""


import pandas as pd
from sklearn import model_selection, tree, metrics
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

FOLD_NUM = 5

# dataframe
df = pd.read_csv("./input/winequality/winequality-red.csv")
quality_map = { 
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 5}
df.loc[:, "quality"] = df.quality.map(quality_map)

# distribution of quality
plt.figure(figsize=(10, 5))
figure = sns.countplot(x="quality", data=df)
figure.set_xlabel("quality", fontsize=20)
figure.set_ylabel("count", fontsize=20)
plt.show()

# stratified cross-validation
# StratifiedKFold: (classification) balance each class in train & val
# GroupKFold: each group only belongs to train or val (split(X, y, group))
# StratifiedGroupKFold: GroupKFold + StratifiedKFold
classifier = tree.DecisionTreeClassifier(max_depth=5)
cols = ['fixed acidity',
        'volatile acidity', 
        'citric acid',
        'residual sugar', 
        'chlorides',
        'free sulfur dioxide', 
        'total sulfur dioxide', 
        'density',
        'pH', 
        'sulphates', 
        'alcohol']
kf = model_selection.StratifiedKFold(n_splits=FOLD_NUM)
avg_val_accuracy = 0
for fold_num, (train_index, val_index) in enumerate(kf.split(X=df, y=df.quality.values)):
    print("fold_num: ", fold_num)
    df_train = df.loc[train_index, :]
    df_val = df.loc[val_index, :]

    # train
    classifier.fit(df_train[cols], df_train.quality)

    # verification
    train_predictions = classifier.predict(df_train[cols])
    val_predictions = classifier.predict(df_val[cols])
    train_accuracy = metrics.accuracy_score(df_train.quality, train_predictions)
    val_accuracy = metrics.accuracy_score(df_val.quality, val_predictions)
    print("train accuracy: {}\nval accuracy: {}".format(train_accuracy, val_accuracy))

    avg_val_accuracy += val_accuracy

print("avg_val_accuracy: ", avg_val_accuracy / FOLD_NUM)