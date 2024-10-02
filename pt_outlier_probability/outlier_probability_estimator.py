from sklearn.base import BaseEstimator, ClassifierMixin,OutlierMixin,check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
import numpy as np
import warnings

class OutlierProbabilityEstimator(BaseEstimator, OutlierMixin):
    
    def __init__(self, outlier_detector = None, 
                 probability_estimator = None,
                 ) -> None:
        """
        Wrapper for sklearn outlier detectors that allows the probability of being an outlier to be calculated.

        Arguments:
        -----------
        outlier_detector  -- Estimator to calculate outlier score. If None use Isolation Forest.
                Default None 
        probability_estimator -- Estimator used to calculate the probability of being an outlier.
                If None use LogisticRegression.  Default None

        """
        super().__init__()

        self.outlier_detector = outlier_detector
        self.probability_estimator = probability_estimator
        



    def fit(self, X, y=None):
        X= self._validate_data(
            X,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
            ensure_2d=True, allow_nd=False,
        )

        self.use_dummy_model_ = False

        self.oultier_detector_ = clone(self.outlier_detector) if self.outlier_detector is not None\
                                    else IsolationForest(random_state=0)
        
        self.oultier_detector_.fit(X,y)
        #TODO offset_ is a part of the interface. Tests checks it
        #I need an estimator that estimates probability, and offset is -0.5
        # decision_function = score_samples - offset_
        self.offset_ = 0.5

        y_pred = self.oultier_detector_.predict(X)
        oultier_val = self.oultier_detector_.decision_function(X)
        oultier_val = oultier_val.reshape(-1,1)

        self.probability_estimator_ = clone(self.probability_estimator) if self.probability_estimator is not None\
                                        else LogisticRegression(random_state=0)

        try:
            self.probability_estimator_.fit(oultier_val, y_pred)
        except Exception as exc:
            warnings.warn("Exception during prob estimator fit. {}".format(exc))
            self.probability_estimator_ = DummyClassifier()
            self.probability_estimator_.fit(oultier_val, y_pred)
            self.use_dummy_model_ = True

        return self

    def predict(self, X):
        """
        Predicts the outcome using outlier detector and probability estimation model.
        Returns:
        --------
        Arrays of zeros and ones
        """
        check_is_fitted(self, ("oultier_detector_", "probability_estimator_", "use_dummy_model_", "offset_"))
        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
            ensure_2d=True, allow_nd=False,
        )


        outlier_vals = self.oultier_detector_.decision_function(X)
        outlier_vals =outlier_vals.reshape(-1,1)

        #TODO for some situations predict alwyays inliners for some sklearn test cases.
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
        check_is_fitted(self, ("oultier_detector_", "probability_estimator_", "use_dummy_model_", "offset_"))
        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
            ensure_2d=True, allow_nd=False,
        )

        if not self.use_dummy_model_:
            outlier_vals = self.oultier_detector_.decision_function(X)
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
    
    #TODO now passes, but what if estimator is used?
    def decision_function(self, X):
        check_is_fitted(self, ("oultier_detector_", "probability_estimator_", "use_dummy_model_", "offset_"))
        decision_values = self.predict_proba(X)[:,1] - self.offset_

        return decision_values
    
    def score_samples(self, X):
        check_is_fitted(self, ("oultier_detector_", "probability_estimator_", "use_dummy_model_", "offset_"))
        return self.predict_proba(X)[:,1]