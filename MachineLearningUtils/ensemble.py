import MachineLearningUtils as mlu
from MachineLearningUtils import utils
import numpy as np
class EnsembleProcessor:
    """
    An Ensemble Processor that supports climb_hill, model_ensemble methods
    """
    def __init__(self, ensemble_mode, metric_str="roc_auc", ensemble_model=None, 
            metrics_score_dict=None, weight_range=np.arange(-1, 1.01, 0.01)):
        self.ensemble_mode = ensemble_mode
        self.ensemble_model = ensemble_model
        self.metric_str = metric_str
        self.metrics_score_dict = metrics_score_dict
        self.weight_range = weight_range
    def find(self, df_model_pred_train, Y_train, df_model_pred_val, Y_val, df_model_pred_test):
        # check ensemble_mode
        if self.ensemble_mode not in mlu.ENSEMBLE_MODE_SET:
            print(f"[Error] : [EnsembleProcessor]: argument: "
                    "ensemble_mode={self.ensemble_mode} is not supported")
            return -1

        if self.ensemble_mode == "climb_hill":
            # check type
            if not isinstance(self.metrics_score_dict, dict):
                print("[Error] : [EnsembleProcessor]: argument: metrics_score_dict "
                        "should be dict type when ensemble_mode is 'climb_hill'")
                return -1
            
            return self._climb_hill_ensemble(self.metrics_score_dict, 
                    df_model_pred_val, Y_val, df_model_pred_test)
        if self.ensemble_mode == "model_ensemble":
            # check type
            if self.ensemble_model is None:
                print("[Error] : [EnsembleProcessor]: argument: ensemble_model "
                        "should not be None when ensemble_mode is 'model_ensemble'")
                return -1

            return self._model_ensemble(df_model_pred_train, Y_train, 
                    df_model_pred_val, Y_val, df_model_pred_test)

    def _model_ensemble(self, df_model_pred_train, Y_train, df_model_pred_val, Y_val, df_model_pred_test):
        self.ensemble_model.fit(df_model_pred_train, Y_train)
        pred_ensemble_val = self.ensemble_mode.predict_proba(df_model_pred_val)
        ensemble_metric_score = mlu.METRICS_DICT[self.metric_str](Y_val, pred_ensemble_val)
        print(f"[Debug] : [model_ensemble] {self.metric_str}: {ensemble_metric_score}")

        pred_test = self.ensemble_mode.predict_proba(df_model_pred_test)
        return pred_test, ensemble_metric_score
        
    def _climb_hill_ensemble(self, metrics_score_dict, df_model_pred_val, Y_val, df_model_pred_test):
        # sort metrics_score_dict
        metrics_score_dict = {k: v for k, v in sorted(
                metrics_score_dict.items(), 
                key=lambda item: item[1], 
                reverse=True)}
        
        # initialize
        list_model_name = list(metrics_score_dict.keys())
        current_best_metric_score = metrics_score_dict[list_model_name[0]]
        current_best_pred_val = df_model_pred_val[list_model_name[0]]
        current_best_pred_test = df_model_pred_test[list_model_name[0]]
        list_model_name.pop(0)
        set_model_name = set(list_model_name)
        could_found_better = True

        # ensemble
        while could_found_better:
            # find combination of model_name & weight
            current_best_model_name, current_best_weight = None, None
            for model_name in set_model_name:
                for weight in self.weight_range:
                    temp_best_pred_val = weight * current_best_pred_val + (1 - weight) * df_model_pred_val[model_name]
                    ensemble_score = mlu.METRICS_DICT[self.metric_str](Y_val, temp_best_pred_val)
                    if ensemble_score >= current_best_metric_score:
                        current_best_metric_score = ensemble_score
                        current_best_model_name = model_name
                        current_best_weight = weight
            # check found better & update 
            if current_best_weight != None:
                current_best_pred_val = current_best_weight * current_best_pred_val + \
                        (1 - current_best_weight) * df_model_pred_val[current_best_model_name]
                current_best_pred_test = current_best_weight * current_best_pred_test + \
                        (1 - current_best_weight) * df_model_pred_test[current_best_model_name]
                print(f"[Debug] : [climb_hill_ensemble] Find better {self.metric_str}: {current_best_metric_score}")
                set_model_name.remove(current_best_model_name)
                if len(set_model_name) == 0:
                    could_found_better = False
            else:
                could_found_better = False
        
        return current_best_pred_test, current_best_metric_score
            