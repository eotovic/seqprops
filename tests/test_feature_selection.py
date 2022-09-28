import unittest
import numpy as np
from seqprops import SequentialPropertiesEncoder
from sklearn.model_selection import StratifiedKFold
from dask.distributed import Client

# Simulates a model whose predictive performance increases
# until seven features are selected, and then it stagnates.
def train_predict_fn(X_train, y_train, X_test):
    nb_instances = X_test.shape[0]
    nb_features = X_test.shape[2] - 1                      # Stop signal is not a feature so we have to subtract one
    predictions = np.zeros(nb_instances)

    fill_end = min(nb_features, 7)
    predictions[:fill_end] = 1
    return predictions

class TestFeatureSelection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dask_client = Client()

    @classmethod
    def tearDownClass(cls):
        cls.dask_client.close()

    # Tests autostop feature with forward feature selection
    def test_autostop_forward(self):
        sequences = np.array(["A"] * 1000)
        y = np.ones(1000)

        encoder = SequentialPropertiesEncoder()
        encoder.feature_selection(
            train_predict_fn = train_predict_fn,
            sequences = sequences,
            y = y,
            nb_features = "auto",
            scoring = "accuracy",
            cv = StratifiedKFold(n_splits=2),
            direction = "forward",
            autostop_patience = 3,
            dask_client = self.dask_client
        )

        nb_selected_features = len(encoder.get_selected_properties())
        self.assertEqual(nb_selected_features, 10)

    # Tests autostop feature with backward feature selection
    def test_autostop_backward(self):
        sequences = np.array(["A"] * 1000)
        y = np.ones(1000)

        encoder = SequentialPropertiesEncoder()
        encoder.feature_selection(
            train_predict_fn = train_predict_fn,
            sequences = sequences,
            y = y,
            nb_features = "auto",
            scoring = "accuracy",
            cv = StratifiedKFold(n_splits=2),
            direction = "backward",
            autostop_patience = 3,
            dask_client = self.dask_client
        )

        nb_selected_features = len(encoder.get_selected_properties())
        expected = len(encoder.get_available_properties()) - 4
        self.assertEqual(nb_selected_features, expected)

    # Tests forward feature selection with specified number of features to select
    def test_predefined_nb_features_forward(self):
        sequences = np.array(["A"] * 1000)
        y = np.ones(1000)

        encoder = SequentialPropertiesEncoder()
        encoder.feature_selection(
            train_predict_fn = train_predict_fn,
            sequences = sequences,
            y = y,
            nb_features = 4,
            scoring = "accuracy",
            cv = StratifiedKFold(n_splits=2),
            direction = "forward",
            dask_client = self.dask_client
        )

        nb_selected_features = len(encoder.get_selected_properties())
        self.assertEqual(nb_selected_features, 4)

    # Tests backward feature selection with specified number of features to select
    def test_predefined_nb_features_backward(self):
        sequences = np.array(["A"] * 1000)
        y = np.ones(1000)

        encoder = SequentialPropertiesEncoder()
        encoder.feature_selection(
            train_predict_fn = train_predict_fn,
            sequences = sequences,
            y = y,
            nb_features = 85,
            scoring = "accuracy",
            cv = StratifiedKFold(n_splits=2),
            direction = "backward",
            dask_client = self.dask_client
        )

        nb_selected_features = len(encoder.get_selected_properties())
        self.assertEqual(nb_selected_features, 85)
