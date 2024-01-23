import unittest
from sklearn.datasets import load_iris
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from ptOutlierProbability.outlier_probability_estimator import OutlierProbabilityEstimator

class EstimatorNBTest(unittest.TestCase):
    
    def get_estimators(self):

        return[
            OutlierProbabilityEstimator(),
            OutlierProbabilityEstimator(outlier_detector_class=LocalOutlierFactor, outlier_detector_arguments={"novelty":True}),
            OutlierProbabilityEstimator(outlier_detector_class=OneClassSVM),
            OutlierProbabilityEstimator(outlier_detector_class=EllipticEnvelope),
        ]


    def test_iris(self):
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0)

        for clf in self.get_estimators():
            clf.fit(X_train,y_train)

            predictions = clf.predict(X_test)

            self.assertIsNotNone(predictions, "Predictions are None!")
            self.assertTrue(len(predictions) == len(X_test), "Wrong number of responses")
            self.assertTrue( np.alltrue( np.in1d(predictions, [-1,1]) ), "Predictions mus be in {-1,1}" )

            prob_predictions = clf.predict_proba(X_test)
            self.assertIsNotNone(prob_predictions, "Proba Predictions are None!")
            self.assertTrue(len(prob_predictions) == len(X_test), "Wrong number of proba responses")
            self.assertTrue( prob_predictions.shape[1] == 2, "Wrong number of probabilities in row.")
            self.assertFalse(  np.any( np.isnan(prob_predictions)), "Nans in proba predictions" )
            self.assertFalse(  np.any( np.isinf(prob_predictions)), "Infinities in proba predictions" )
            self.assertTrue( np.all( prob_predictions <=1.0), "Some probas above one")
            self.assertTrue( np.all( prob_predictions >=0.0), "Some probas below zero")

            rowsums = np.sum(prob_predictions, axis=1)
            self.assertTrue(np.allclose(rowsums,1.0), "Some probs do not sum to one")

            clf.fit_predict(X_train)

    def test_pipeline(self):
        X, y = load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0)

        for clf in self.get_estimators():

            pipe = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])

            pipe.fit(X_train,y_train)

            predictions = clf.predict(X_test)

            self.assertIsNotNone(predictions, "Predictions are None!")
            self.assertTrue(len(predictions) == len(X_test), "Wrong number of responses")
            self.assertTrue( np.alltrue( np.in1d(predictions, [-1,1]) ), "Predictions mus be in {-1,1}" )

            prob_predictions = pipe.predict_proba(X_test)
            self.assertIsNotNone(prob_predictions, "Proba Predictions are None!")
            self.assertTrue(len(prob_predictions) == len(X_test), "Wrong number of proba responses")
            self.assertTrue( prob_predictions.shape[1] == 2, "Wrong number of probabilities in row.")
            self.assertFalse(  np.any( np.isnan(prob_predictions)), "Nans in proba predictions" )
            self.assertFalse(  np.any( np.isinf(prob_predictions)), "Infinities in proba predictions" )
            self.assertTrue( np.all( prob_predictions <=1.0), "Some probas above one")
            self.assertTrue( np.all( prob_predictions >=0.0), "Some probas below zero")

            rowsums = np.sum(prob_predictions, axis=1)
            self.assertTrue(np.allclose(rowsums,1.0), "Some probs do not sum to one")

            clf.fit_predict(X_train)

if __name__ == '__main__':
    unittest.main()