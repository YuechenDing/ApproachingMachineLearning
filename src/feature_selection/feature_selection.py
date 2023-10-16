"""
Feature selection:
    - Statistical parameters:
        - remove low variance features (feature_selection.VarianceThreshold(threshold))
        - high correlation features (df.corr())
    - Univariate feature selection (a scoring of each feature against target):
        - Mutual information, ANOVA F-test and chi2 (SelectKBest, SelectPercentile)
    - Greedy feature selection:
        - Add: add one feature each loop to check whether it improves the evaluation score
        - Eliminate: remove one feature each loop(linear model: coefficients; tree: importance) 
          (feature_selection.RFE(estimator, n_feature_to_select); tree_model.feature_importances_;
           feature_selection.SelectFromModel(estimator))

"""