"""
Time Series:
    - Records data: 
        - aggregated features: aggregate feature characteristics on each id
    - Series list data:
        - statistical features: Mean, Max, Min, Unique, Skew, Kurtosis, Kstat, Percentile, Quantile (tsfresh)
Continuous:
    - Polynomial features: preprocessing.PolynomialFeatures(degree)
    - Binning: Cut Continuous feature into discrete bins (pd.cut(df[feature], bins))
    - Log transform: Reduce variance of the feature (df.feature.apply(lambda x: np.log(1+x)))
    - Fill missing/NaN (Tree models can handle missing values themselves):
        - fill with constance, median, mean, ...
        - KNN: replace NaN with KNN (sklearn.impute.KNNImputer(n_neighbors))
        - Train a regression model based on other features

"""