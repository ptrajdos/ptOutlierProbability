import unittest
from sklearn.datasets import load_iris
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from pt_outlier_probability.outlier_probability_estimator import OutlierProbabilityEstimator

class OutlierEstimatorTest(unittest.TestCase):
    
    def get_estimators(self):

        return[
            OutlierProbabilityEstimator(),
            OutlierProbabilityEstimator(outlier_detector=LocalOutlierFactor(novelty=True)),
            OutlierProbabilityEstimator(outlier_detector=OneClassSVM()),
            OutlierProbabilityEstimator(outlier_detector=EllipticEnvelope()),
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

    def test_explicit_outliers(self):

        X_train = np.random.normal(size=(1000,2))

        X_test_out = np.random.random((1000,2)) + (10,10)

        for clf in self.get_estimators():
            clf.fit(X_train, None)

            y_pred_out = clf.predict(X_test_out)

            pred_out_des = np.zeros_like(y_pred_out)
            pred_out_des[:] = -1

            self.assertTrue(np.allclose(y_pred_out, pred_out_des), "Clear outliers shoule all be predicted as -1!")

            y_proba_out = clf.predict_proba(X_test_out)

            for proba in y_proba_out:
                self.assertTrue(proba[0] > proba[1], "Soft predictions should indicate an outlier!")

    def test_only_inliers(self):

        X_train = np.random.normal(loc=0, scale=0.0005, size=(1000,2))
        X_test = np.random.normal(loc=0, scale=0.0005, size=(1000,2))

        for clf in self.get_estimators():
            clf.fit(X_train, None)

            predictions = clf.predict(X_test)

            self.assertIsNotNone(predictions, "Predictions are None!")
            self.assertTrue(len(predictions) == len(X_test), "Wrong number of responses")
            self.assertTrue( all( np.in1d(predictions, [-1,1]) ), "Predictions mus be in {-1,1}" )

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

    def test_uniform(self):

        X_train = np.random.random( (100,2) )
        X_test = np.random.random( (100,2) )

        for clf in self.get_estimators():
            clf.fit(X_train, None)

            predictions = clf.predict(X_test)

            self.assertIsNotNone(predictions, "Predictions are None!")
            self.assertTrue(len(predictions) == len(X_test), "Wrong number of responses")
            self.assertTrue( all( np.in1d(predictions, [-1,1]) ), "Predictions mus be in {-1,1}" )

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

    def test_grid_search(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=0)

        rf = IsolationForest()
        svm = OneClassSVM()

        clf = OutlierProbabilityEstimator()

        pipe = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])

        params_grid = [{
                    'classifier__outlier_detector': [rf],
                    'classifier__outlier_detector__n_estimators': [2,3]

                },
                {
                    'classifier__outlier_detector': [svm],
                    'classifier__outlier_detector__nu': [0.1,0.2]
                }]

        grd = GridSearchCV(pipe, param_grid=params_grid)

        grd.fit(X_train, y_train)
        predictions = grd.predict(X_test)

        self.assertIsNotNone(predictions, "Predictions are None!")
        self.assertTrue(len(predictions) == len(X_test), "Wrong number of responses") 


if __name__ == '__main__':
    unittest.main()