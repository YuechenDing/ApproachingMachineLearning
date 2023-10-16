"""
Ensemble(Stacking):
    - (1)stacking with parameter for each model; (2)stacking with another model
    - mean_probs, max_voting, mean_of_ranks
    - process:
        - create k-fold cross validation
        - get pred_probe on each model on each fold
        - (1)construct partial functions <--> ensemble_parameters, optimize for hyperparameter-tuning
        - (2)use model predictions on the same cross-validation set to train another model

"""