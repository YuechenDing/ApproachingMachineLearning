"""
Classification:
    - Accuracy is used in balanced class situations
    - unbalanced 2-class: TP, FP, TN, FN -- P, R, PR-Curve, F1 -- TPR, FPR, ROC-Curve, AUC -- log loss(Cross Entropy)
        - ROC-Curve is better than PR-Curve, ROC-Curve is much more stable than PR-Curve (FPR is stable, P is not stable enough)
        - ROC-Curve can be used to choose classification threshold
        - AUC: a random positive sample ranks higher than a random negative sample with the probability of AUC score
    - multi-class: macro/micro/weighted P/R
Regression: 
    - MSE, RMSE, R-Squared, quadratic weighted kappa, Matthew’s Correlation Coefficient
"""