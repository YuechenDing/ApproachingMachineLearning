"""
Grid search:
    - set a list of values for each hyperparameter, trained on each combination
    - model_selection.GridSearchCV(estimator, param_grid, scoring, verbose, n_jobs, cv)
Random search:
    - select n_iter random combination
    - model_selection.RandomizedSearch(estimator, param_grid, n_iter, scoring, verbose, n_jobs, cv)
Minimization of functions:
    - Function optimization + search method(param_space -> partial_function -> optim_method)

"""