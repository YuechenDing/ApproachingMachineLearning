"""
Decision Tree + simple validation

"""

import pandas as pd
from sklearn import tree, metrics
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

TRAIN_VAL_SPLIT_RATE = 0.7

# df read csv
df = pd.read_csv("./input/winequality/winequality-red.csv")
print("-"*20 + "\nData frame:\n" + "-"*20)
print(df)

# mapping value
quality_map = { 
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 5}
df.loc[:, "quality"] = df.quality.map(quality_map)  # 2 ways of accessing column Series

# df dataset train/val
df = df.sample(frac=1).reset_index(drop=True)  # sample(frac, axis), reset_index
df_train = df.head(int(TRAIN_VAL_SPLIT_RATE * len(df)))
df_val = df.tail(len(df) - int(TRAIN_VAL_SPLIT_RATE * len(df)))

train_accuracy_list = [0.5]
val_accuracy_list = [0.5]
# DT model
for depth in range(1, 25):
    classifier = tree.DecisionTreeClassifier(max_depth=depth)
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
    classifier.fit(df_train[cols], df_train.quality)  # select dataframe by list

    # verification
    train_predictions = classifier.predict(df_train[cols])
    val_predictions = classifier.predict(df_val[cols])
    train_accuracy_list.append(metrics.accuracy_score(df_train.quality, train_predictions))
    val_accuracy_list.append(metrics.accuracy_score(df_val.quality, val_predictions))

plt.figure(figsize=(10, 5))   # figure -> style -> plot -> x/yticks -> legend, x/ylabel
sns.set_style("whitegrid")
plt.plot(train_accuracy_list, label="train accuracy")
plt.plot(val_accuracy_list, label="val accuracy")
plt.legend(loc="upper left", prop={"size": 15})
plt.xticks(range(0, 26, 5))
plt.xlabel("max_depth", size=20)
plt.ylabel("accuracy", size=20)
plt.show()