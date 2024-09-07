# functions to tune hyperparameters
import pandas as pd
from typing import Callable
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from defs import VAL_SCORE, EXP_WIDGET, CLF_WIDGET, ACCURACY, F1_SCORE, AUC, METRIC_WIDGET, \
    LM_WIDGET, TEST_SCORE
from ml import CLF_MAP, METHOD_MAP
from visualizing import plot_scores

METRIC_MAP = {ACCURACY: accuracy_score, F1_SCORE: f1_score, AUC: roc_auc_score}


def tune_param(Ssl_clf, Clf, feature_df: pd.DataFrame, score_fcn: Callable, target: str, param_tune: dict, 
               label_sizes: list, n_workers: int=1) -> pd.DataFrame:
    """
    tune hyperparameters for different label sizes
    @param Ssl_clf: semi supervised learning classifier to be tuned
    @param Clf: classifier class
    @param feature_df: dataframe with features
    @param score_fcn: function to score classifier
    @param target: target variable to be predicted
    @param param_tune: dictionary with parameters to be tuned with parameter name as key and list of parameter values as value
    @param label_sizes: list of label sizes
    @param n_workers: number of parallel workers
    @return: dataframe with scores for each parameter value and training size
    """
    def init_and_score(param: dict) -> dict:
        label_size = param.pop('label_size')
        ssl_clf = Ssl_clf(feature_df, Clf, score_fcn, target, **param)
        return ssl_clf.fit_score(label_size)
    
    scores = []
    best_param = {label_size: dict() for label_size in label_sizes}
    for param, param_vals in param_tune.items():
        # check if param_vals is list
        if not isinstance(param_vals, list):
            # convert to list
            param_vals = [param_vals]
        if  len(param_vals) == 1:
            # set best parameter for each training size
            for label_size in label_sizes:
                best_param[label_size][param] = param_vals[0]
            # skip tuning if only one parameter value
            continue
        
        print(f'Tuning {param}')
        input_dicts = []
        for param_val in param_vals:
            for label_size in label_sizes:
                add_dict = {param: param_val, 'label_size': label_size}
                add_dict.update(best_param[label_size])
                input_dicts.append(add_dict)
            
        results = Parallel(n_jobs=n_workers)(delayed(init_and_score)(input_dict) for input_dict in input_dicts)
        # append name of parameter that was tuned
        [dic.update({'param_tuned': param}) for dic in results]
        scores.extend(results)

        # get best parameter for each training size
        for param_score in scores:
            label_size = param_score['label_size']
            if param_score[VAL_SCORE] > best_param[label_size].get(param, -1):
                best_param[label_size][param] = param_score[param]

    # check if parameter dict is empty
    if not param_tune:
        print('No parameters to tune')
        # calculate scores for each training size
        input_dicts = [{'label_size': label_size} for label_size in label_sizes]
        results = Parallel(n_jobs=n_workers)(delayed(init_and_score)(input_dict) for input_dict in input_dicts)
        scores.extend(results)
    
    return pd.DataFrame(scores)


class ExperimentHandler:

    def __init__(self, feature_df: pd.DataFrame, target: str):
        # list of experiment names
        self.exp_names = []
        # map experiment name to results
        self.results_df = pd.DataFrame()
        # map experiment name to parameters
        self.params = dict()
        # map experiment name to best parameters
        self.best_scores_df = pd.DataFrame()

        self.feature_df = feature_df
        self.target = target

    def add_experiment(self, exp_name:str, clf_name:str, metric_name:str, method_name:str, tuning_dict:dict, 
                       plot: bool=True):
        """
        add experiment to handler
        @param exp_name: name of experiment
        @param clf_name: name of classifier
        @param metric_name: name of metric
        @param method_name: name of learning method
        @param tuning_dict: dictionary with parameters to be tuned with parameter name as key and list of 
                            parameter values as value
        @param plot: if True, plot scores for each parameter value and training size
        """
        # check that experiment name is unique
        if exp_name in self.exp_names:
            raise ValueError(f'Experiment name {exp_name} already exists')
        
        # get classifier class
        Clf_class = CLF_MAP[clf_name]

        # get metric
        metric_fcn = METRIC_MAP[metric_name]

        # get learning method
        Ssl_clf = METHOD_MAP[method_name]

        # get training sizes
        label_sizes = tuning_dict.pop('label_sizes')

        # tune parameters
        tune_df = tune_param(Ssl_clf, Clf_class, self.feature_df, metric_fcn, self.target, tuning_dict, 
                             label_sizes)
        
        # assign experiment, classifier, metric, and learning method to dataframe
        tune_df[EXP_WIDGET] = exp_name
        tune_df[CLF_WIDGET] = clf_name
        tune_df[METRIC_WIDGET] = metric_name
        tune_df[LM_WIDGET] = method_name

        # get best scores
        best_scores_df = get_best_score(tune_df)

        # add dataframe to results
        self.exp_names.append(exp_name)
        self.params[exp_name] = tuning_dict        
        self.results_df = pd.concat([self.results_df, tune_df], ignore_index=True)
        self.best_scores_df = pd.concat([self.best_scores_df, best_scores_df], ignore_index=True)

        figs = []
        if plot:
            # iterate through all parameters
            for param, param_vals in tuning_dict.items():
                # check if multiple parameter values
                if isinstance(param_vals, list) and len(param_vals) > 1:
                    plot_df = tune_df.loc[tune_df['param_tuned'] == param]
                    figs.append(plot_scores(plot_df, 'label_size', TEST_SCORE, param))
        
        return figs


def get_best_score(score_df:pd.DataFrame, score_type: str=VAL_SCORE) -> pd.DataFrame:
    """
    get row of best score for each training size
    @param score_df: dataframe with scores
    @param score_type: type of score to be considered
    @return: dataframe with best scores for each training size
    """
    best_df = (score_df
               .groupby('label_size')
               .apply(lambda x: x.loc[x[score_type].idxmax()])
               .reset_index(drop=True))
    return best_df
