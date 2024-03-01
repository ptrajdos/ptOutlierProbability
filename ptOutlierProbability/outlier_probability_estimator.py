from sklearn.base import BaseEstimator, ClassifierMixin,check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
import numpy as np
import warnings

class OutlierProbabilityEstimator(BaseEstimator, ClassifierMixin):
    
    def __init__(self, outlier_detector = IsolationForest(), 
                 probability_estimator = LogisticRegression(),
                 ) -> None:
        """
        Wrapper for sklearn outlier detectors that allows the probability of being an outlier to be calculated.

        Arguments:
        -----------
        outlier_detector  -- estimator to calculate outlier score 
        probability_estimator -- estimator used to calculate the probability of being an outlier, 

        """
        super().__init__()

        self.outlier_detector = outlier_detector
        self.probability_estimator = probability_estimator
        self.use_dummy_model = False



    def fit(self, X, y=None):

        self.oultier_detector_ = clone(self.outlier_detector)
        self.oultier_detector_.fit(X,y)

        y_pred = self.oultier_detector_.predict(X)
        oultier_val = self.oultier_detector_.score_samples(X)
        oultier_val = oultier_val.reshape(-1,1)

        self.probability_estimator_ = clone(self.probability_estimator)

        try:
            self.probability_estimator_.fit(oultier_val, y_pred)
        except Exception as exc:
            warnings.warn("Exception during prob estimator fit. {}".format(exc))
            self.probability_estimator_ = DummyClassifier()
            self.probability_estimator_.fit(oultier_val, y_pred)
            self.use_dummy_model = True

        return self

    def predict(self, X):
        """
        Predicts the outcome using outlier detector and probability estimation model.
        Returns:
        --------
        Arrays of zeros and ones
        """
        check_is_fitted(self, ("oultier_detector_", "probability_estimator_"))

        outlier_vals = self.oultier_detector_.score_samples(X)
        outlier_vals =outlier_vals.reshape(-1,1)

        predictions = self.probability_estimator_.predict(outlier_vals)

        return predictions

    def predict_proba(self,X):
        """
        Predict points probability of being an outlier

        Returns:
        -------
        numpy ndarray of two columns.
        The first column contains probability of being an outlier
        The second column contains probability of not being an outlier.
        """
        check_is_fitted(self, ("oultier_detector_", "probability_estimator_"))

        if not self.use_dummy_model:
            outlier_vals = self.oultier_detector_.score_samples(X)
            outlier_vals =outlier_vals.reshape(-1,1)

            prob_predictions = self.probability_estimator_.predict_proba(outlier_vals)

            return prob_predictions
        
        preds = self.probability_estimator_.predict(X)
        prob_predictions = np.zeros((len(preds),2))
        if preds[0] == -1:
            prob_predictions[:,0] = 1
        else:
            prob_predictions[:,1] = 1

        return prob_predictions
        

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)
    
    def _more_tags(self):
        return {
            "_xfail_checks":{
                "check_parameters_default_constructible":
                    "transformer has 1 mandatory parameter",
            }
        }