## Pandas

# **  data overview  **
# Dataframe.describe()
# DataFrame[column].value_counts()

# **  indexing  **
# DataFrame.loc
# DataFrame.iloc
# DataFrame.column
# DataFrame[column]; DataFrame[list_like_column]
# DataFrame.head(int); DataFrame.tail(int)

# **  delete columns/rows  **
# DataFrame.drop(labels=None, *, axis=0, index=None, columns=None, ...)


## Sklearn
# **  Categorical Encodings  **
# encoder = sklearn.preprocessing.XXXEncoder(*, 
#       handle_unknown='ignore', drop='first'|'if_binary', sparse_output=True)
# encoder.fit(X)
# encoder.transform(X)   ->  array
# encoder.fit_transform(X)    ->  array
# encoder.get_feature_names_out(list_like_features)    ->  array_name