# ML classes
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from typing import Callable, Sequence
from pympler import asizeof
import time

from defs import TRAIN, VAL, TEST, TRAIN_SCORE, VAL_SCORE, TEST_SCORE, XGB_CLF, RF_CLF, SVM_CLF, BASE_METHOD, ST_METHOD, LP_METHOD, LS_METHOD

# map classifier names to classes
CLF_MAP = {XGB_CLF: xgb.XGBClassifier, RF_CLF: RandomForestClassifier, SVM_CLF: SVC}


class BaselineClf:
    """
    Classifier without semi-supervised learning
    """
    model = 'Baseline'

    def __init__(self, feature_df: pd.DataFrame, Clf_class: BaseEstimator, score_fcn: Callable, target_col: str):
        """
        initialize base classifier
        @param feature_df: dataframe with features and target split into TRAIN, VAL, and TEST sets
        @param Clf_class: classifier class
        @param score_fcn: scoring function
        @param target_col: target column name
        """
        self.clf = Clf_class()
        self.feature_df = feature_df
        self.score_fcn = score_fcn
        self.target_col = target_col
        self.label_size = None
    
    def fit(self, label_size: int):
        """
        fit classifier on label_size samples of training data
        @param label_size: number of samples to fit
        """
        self.label_size = label_size
        train_df = self.feature_df.loc[TRAIN].drop(columns=self.target_col).head(label_size)
        target = self.feature_df.loc[TRAIN, self.target_col].head(label_size)
        self.clf.fit(train_df, target)
        return self
    
    def predict(self, feature_df: pd.DataFrame) -> Sequence:
        """
        make predictions
        @param feature_df: dataframe with features
        @return: predictions
        """
        return self.clf.predict(feature_df)

    def score(self) -> dict:
        """
        score training, validation, and test data predictions, append parameters and model size
        @return: dictionary with scores and parameters
        """
        # add parameters
        score_dict = self.get_parameters()

        # training error
        train_df = self.feature_df.loc[TRAIN].drop(columns=self.target_col).head(self.label_size)
        train_target = self.feature_df.loc[TRAIN, self.target_col].head(self.label_size)
        score_dict[TRAIN_SCORE] = self.score_fcn(train_target, self.predict(train_df))

        # validation and test error
        for data_set, score_key in zip([VAL, TEST], [VAL_SCORE, TEST_SCORE]):
            df = self.feature_df.loc[data_set].drop(columns=self.target_col)
            target = self.feature_df.loc[data_set, self.target_col]
            score_dict[score_key] = self.score_fcn(target, self.predict(df))

        return score_dict
    
    def get_parameters(self) -> dict:
        """
        get model name and label_size
        @return: dictionary with model name and label_size
        """
        return {'model': self.model, 'label_size': self.label_size}
    
    def get_model_size(self) -> int:
        """
        get size of classification model
        @return: size of model in bytes
        """
        return asizeof.asizeof(self.clf)
    
    def fit_score(self, label_size: int) -> dict:
        """
        fit classifier and score predictions
        @param label_size: number of samples to fit
        @return: dictionary with scores, parameters, times to fit and score, and model size
        """
        # get time to fit and score
        start = time.time()
        _ = self.fit(label_size)
        fit_time = time.time() - start

        # score
        start = time.time()
        score_dict = self.score()
        score_time = time.time() - start

        score_dict['fit_time'] = fit_time
        score_dict['score_time'] = score_time

        # get memory usage
        score_dict['model_size'] = self.get_model_size()
        return score_dict
    

class SelfTrainingClf(BaselineClf):
    """
    Classifier with self-training
    """
    model = 'SelfTraining'

    def __init__(self, feature_df: pd.DataFrame, Clf_class: BaseEstimator, score_fcn: Callable, target_col: str, 
                 threshold: float=0.75, criterion: str='threshold', k_best: int=10):
        """
        initialize classifier with self-training
        @param feature_df: dataframe with features and target split into TRAIN, VAL, and TEST sets
        @param Clf_class: classifier class
        @param score_fcn: scoring function
        @param target_col: target column name
        @param threshold: threshold for self-training
        @param criterion: criterion for self-training
        @param k_best: number of best predictions to use for self-training
        """
        super().__init__(feature_df, Clf_class, score_fcn, target_col)
        
        self.clf = SelfTrainingClassifier(Clf_class(), threshold=threshold, criterion=criterion, k_best=k_best)
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.label_size = None

    def fit(self, label_size: int):
        """
        fit classifier on label_size samples of training data, unlabel targets exceeding label_size
        @param label_size: number of samples to fit
        """
        self.label_size = label_size
        train_df = self.feature_df.loc[TRAIN].drop(columns=self.target_col)

        target = self.feature_df.loc[TRAIN, self.target_col].copy()
        # set the labels of the unlabelled data to -1
        if label_size < len(train_df):
            target.values[label_size:] = -1
        
        self.clf.fit(train_df, target)
        return self

    def get_parameters(self) -> dict:
        """
        get model name, label_size, and self-training parameters
        """
        return {**super().get_parameters(), 'threshold': self.threshold, 'criterion': self.criterion, 
                'k_best': self.k_best}
    

class LabelPropagationClf(BaselineClf):
    """
    classifier with label propagation
    """

    model = 'LabelPropagation'

    def __init__(self, feature_df: pd.DataFrame, Clf_class: BaseEstimator, score_fcn: Callable, target_col: int, kernel: str='rbf', 
                 gamma: float=20.0, n_neighbors: int=7, rbf_size: int=3000):
        """
        initialize classifier with label propagation
        @param feature_df: dataframe with features and target split into TRAIN, VAL, and TEST sets
        @param Clf_class: classifier class
        @param score_fcn: scoring function
        @param target_col: target column name
        @param kernel: kernel for label propagation
        @param gamma: gamma for label propagation
        @param n_neighbors: number of neighbors for label propagation
        @param rbf_size: size of sample to fit rbf kernel
        """
        super().__init__(feature_df, Clf_class, score_fcn, target_col)
        self.clf = Clf_class()
        self.labeler = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors)
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.rbf_size = rbf_size
        self.label_size = None

    def fit(self, label_size: int):
        """
        unlabel targets beyond label_size, fit label propagation (only use self.rbf_size samples for rbf kernel), 
        predict labels of unlabeled targets, fit classifier
        """
        # get training data and target
        self.label_size = label_size
        train_df = self.feature_df.loc[TRAIN].drop(columns=self.target_col)
        target = self.feature_df.loc[TRAIN, self.target_col].copy()

        # set the labels of the unlabelled data to -1
        label_size = min(label_size, len(train_df))
        target.values[label_size:] = -1

        if self.kernel == 'rbf':
            # fit on rbf sample size
            n_sample = min(self.rbf_size, label_size)
            rbf_df = train_df.head(n_sample)
            rbf_target = target.head(n_sample)
            _ = self.labeler.fit(rbf_df, rbf_target)
        else:
            # fit on all samples
            _ = self.labeler.fit(train_df, target)

        labels_fit = self.get_fit_labels(label_size, train_df, target)

        # fit classifier
        self.clf.fit(train_df, labels_fit)

        return self
    
    def get_fit_labels(self, label_size: int, train_df: pd.DataFrame, target: pd.Series) -> np.ndarray:
        """
        get labels to fit classifier
        @param label_size: number of samples to fit
        @param train_df: training dataframe
        @param target: target series
        @return: labels to fit classifier
        """
        # get labels for the training set
        if label_size < len(train_df):
            labels_pred = self.labeler.predict(train_df.iloc[label_size:])
            return np.concatenate([target.head(label_size).values, labels_pred])
        return target.values
  
    def get_model_size(self) -> int:
        """
        calculate model size as sum of label propagation and classifier sizes
        @return: size of model in bytes
        """
        # get size of label propagation model and classifier
        return asizeof.asizeof(self.labeler) + asizeof.asizeof(self.clf)
    
    def get_parameters(self) -> dict:
        """
        get model name, label_size, and label propagation parameters
        """
        return {**super().get_parameters(), 'kernel': self.kernel, 'gamma': self.gamma, 
                'n_neighbors': self.n_neighbors, 'rbf_size': self.rbf_size}


class LabelSpreadingClf(LabelPropagationClf):
    """
    classifier with label spreading
    """
    model = 'LabelSpreading'

    def __init__(self, feature_df: pd.DataFrame, Clf_class: BaseEstimator, score_fcn: Callable, target_col: str, kernel: str='rbf', 
                 gamma: float=20, n_neighbors: int=7, rbf_size: int=3000, alpha: float=0.2):
        """
        initialize classifier with label spreading
        @param feature_df: dataframe with features and target split into TRAIN, VAL, and TEST sets
        @param Clf_class: classifier class
        @param score_fcn: scoring function
        @param target_col: target column name
        @param kernel: kernel for label spreading
        @param gamma: gamma for label spreading
        @param n_neighbors: number of neighbors for label spreading
        @param rbf_size: size of sample to fit rbf kernel
        @param alpha: alpha for label spreading
        """
        super().__init__(feature_df, Clf_class, score_fcn, target_col, kernel, gamma, n_neighbors, rbf_size)
        self.labeler = LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha)
        self.alpha = alpha

    def get_fit_labels(self, _, train_df, __) -> np.ndarray:
        """
        set all labels to predictions of labeler
        @param _: not used
        @param train_df: training dataframe
        @param __: not used
        @return: labels to fit classifier
        """
        return self.labeler.predict(train_df)
    
    def get_parameters(self) -> dict:
        """
        get model name, label_size, and label spreading parameters
        @return: dictionary with model name, label_size, and label spreading parameters
        """
        return {**super().get_parameters(), 'alpha': self.alpha}  
    
# map learning method names to classes
METHOD_MAP = {BASE_METHOD: BaselineClf, ST_METHOD: SelfTrainingClf, LP_METHOD: LabelPropagationClf, LS_METHOD: LabelSpreadingClf}
