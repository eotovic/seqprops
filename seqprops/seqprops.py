import numpy as np
import pandas as pd
from io import BytesIO
import importlib
from sklearn.metrics import get_scorer

class SequentialPropertiesEncoder:
    def __init__(self, stop_signal=True, max_seq_len=None, selected_properties=None, scaler=None):
        """
        Parameters
        ----------
        stop_signal (bool) : Whether to use stop signal or not. By default, stop signal is used.
        max_seq_len (int) : The length to which all sequences are padded. If not specified, it is automatically inferred.
        selected_properties (list) : List of property names. Limits feature set to the specified properties. 
        scaler : Scaler from scikit-learn (e.g. MinMaxScaler) that will be used for feature scaling. No scaling is applied if not specified.
        """
        self.stop_signal = stop_signal
        self.max_seq_len = max_seq_len
        self.mappings = None
        self.selected_properties = None
        self.mask_value = None

        # Preload properties from CSV file
        bytes = importlib.resources.read_binary("seqprops.data", "aadata.csv")
        self.properties_data = pd.read_csv(BytesIO(bytes))

        # Scale properties
        if scaler is not None:
            self.properties_data[self.get_available_properties()] = scaler.fit_transform(self.properties_data[self.get_available_properties()])
        
        # Create mappings
        if selected_properties is None:
            self.select_properties(self.get_available_properties())
        else:
            self.select_properties(selected_properties)

    def select_properties(self, properties):
        """
        Select a subset of properties for encoding.

        Parameters
        ----------
        properties (list) : List of property names.
        """
        self.selected_properties = list(properties)
        self.mappings = {}
        selected_properties_data = self.properties_data[properties]

        for row_idx, amino_acid in enumerate(self.properties_data["AminoAcid"]):
            amino_acid_properties = selected_properties_data.iloc[row_idx].to_list()
            self.mappings[amino_acid] = amino_acid_properties

    def get_selected_properties(self):
        """
        Returns a list of currently selected properties for encoding. 

        Returns
        ----------
        selected_properties (list) : List of property names.
        """
        return list(self.selected_properties)

    def get_available_properties(self):
        """
        Returns a list of all available properties. 

        Returns
        ----------
        selected_properties (list) : List of property names.
        """
        columns = list(self.properties_data.columns)
        columns.remove("AminoAcid")
        return columns

    def encode(self, sequences):
        """
        Encodes the list of given sequences. If max_seq_len was not specified, all 
        sequences are padded to the length of the longest sequence, otherwise they
        are padded to the max_seq_len.

        Parameters
        ----------
        sequences (list) : List of sequences. Each sequence should be represented
                           as a string with one letter amino acid codes.

        Returns
        ----------
        encoded_sequences (numpy.ndarray) : Array containing encoded sequences.
        """
        encoded_sequences = []
        max_seq_len = 0
        for sequence in sequences:
            sequence = sequence.upper()
            encoded_sequence = []
            for amino_acid in sequence:
                vec = list(self.mappings[amino_acid])
                if self.stop_signal:
                    vec = vec + [0]
                encoded_sequence.append(vec)
            encoded_sequences.append(encoded_sequence)
            max_seq_len = max(max_seq_len, len(sequence))
        
        if self.max_seq_len is not None:
            max_seq_len = self.max_seq_len
        max_seq_len += 1

        for sequence in encoded_sequences:
            vec_length = len(sequence[0])
            while len(sequence) < max_seq_len:
                vec = [0 for _ in range(vec_length)]
                if self.stop_signal:
                    vec[-1] = 1
                sequence.append(vec)

        return np.array(encoded_sequences)

    def _train_and_predict(self, train_predict_fn, X, y, current_features, train_idx, test_idx):
        """
        Internal function used to train a model on a train set with a subset 
        of features and then make the prediction on a test set.

        Parameters
        ----------
        train_predict_fn (function) : User-defined function which takes arguments (X_train, y_train, X_test) 
                                      and returns a predictions for X_test
        X (ndarray) : Data inputs.
        y (ndarray) : Data outputs.
        current_features (list) : List of property indices that should be used in evaluation. 
        train_idx (list) : List of indices that should be used for training.
        test_idx (list) : List of indices that should be used for testing.

        Returns
        ----------
        Returns a pair (y_test, y_pred)
        """
        X = X[:, :, current_features]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        y_pred = train_predict_fn(X_train, y_train, X_test)
        print("Predict done {}".format(X.shape))
        return (y_test, y_pred)

    def _compute_cross_val_score(self, scorer, test_outputs):
        """
        Internal function used to compute cross-validation score.

        Parameters
        ----------
        scorer (scikit-learn scorer) : Scorer object from scikit-learn. 
                                       See https://scikit-learn.org/stable/modules/model_evaluation.html 
                                       for more information.
        test_outputs (list) : List of pairs. Each pair should be of type (y_true, y_pred)

        Returns
        ----------
        Returns cross-validation score.
        """
        scores = [scorer._score_func(y_true, y_pred) for y_true, y_pred in test_outputs]
        return np.mean(scores)


    # Scoring: str or create your own with make_scorer. Uvijek se maksimizira.
    # score_calculation_method: može biti korisno kada se koristi LeaveOneOut (cross_val_concat)   score_calculation_method="cross_val_mean", 
    def feature_selection(self, train_predict_fn, sequences, y, nb_features, scoring, cv, direction='forward', autostop_patience=3, dask_client=None):
        """
        Performs feature selection using Sequential Feature Selection algorithm.
        If the number of features is specified, algorithm stops when the 
        defined number of features has been selected. In the case of automatic
        stopping, it stops when there was no increase in score for defined number
        of iterations. Nested cross-validation is used to evaluate each feature.

        Once feature selection if done, select_properties() is automatically called
        with optimal feature set. Selected features can be retrieved
        with get_selected_properties().

        Computation if executed in parallel using Dask backend.

        Parameters
        ----------
        train_predict_fn (function) : User-defined function which takes arguments (X_train, y_train, X_test) 
                                      and returns a predictions for X_test
        sequences (list) : List of sequences. Each sequence should be represented
                           as a string with one letter amino acid codes.
        y (ndarray) : Outputs for given sequences.
        nb_features (int or "auto") : Specify a fixed number of features to select.
                                      Specify "auto" if you would like to automatically stop feature
                                      selection when score has not increased for autostop_patience iterations.
        scoring (str or scikit-learn's scorer) : Score which should be used to evaluate the features (e.g. 'roc_auc')
                                                 or make your own scorer with make_scorer function.
        cv : Cross validation for feature evaluation (e.g. StratifiedKFold).
        direction (str) : Algorithm starts with empty feature set which is expanded each iteration if this 
                          argument is set to "forward". Algorithm starts with all available features and 
                          at each iteration removes the worst feature from feature set if this argument 
                          is set to "backward". Depending on the number of features, one may be faster
                          or slower.
        dask_client : Dask client which will be used for paralellization.
        

        Returns
        ----------
        Returns list containing the score and a list of used features for each iteration.
        """
        all_features = self.get_available_properties()

        if isinstance(scoring, str):
            scorer = get_scorer(scoring)

        all_features = self.get_available_properties()
        features_to_consider = list(range(len(all_features)))
        stop_signal_idx = len(all_features)
        if direction == 'forward':
            selected_features = []
        elif direction == 'backward':
            selected_features = list(range(len(all_features)))

        history = []
        self.select_properties(self.get_available_properties())
        X = self.encode(sequences)
        X_future = dask_client.scatter(X, broadcast=True)
        y_future = dask_client.scatter(y, broadcast=True)

        autostop_maximum = 0
        autostop_counter = 0
        while len(selected_features) != nb_features:    
            scores = []
            for feature_idx in features_to_consider:
                current_features = list(selected_features)

                if direction == 'forward':
                    current_features.append(feature_idx)
                elif direction == 'backward':
                    current_features.remove(feature_idx)

                if self.stop_signal:
                    current_features.append(stop_signal_idx)
                
                cross_val_outputs = []
                for train_idx, test_idx in cv.split(X, y):
                    cross_val_outputs.append(dask_client.submit(self._train_and_predict, train_predict_fn, X_future, y_future, current_features, train_idx, test_idx, pure=False))
                scores.append(dask_client.submit(self._compute_cross_val_score, scorer, cross_val_outputs, pure=False))
           
            scores = dask_client.gather(scores)
            best_idx = np.argmax(scores)
            best_feature_idx = features_to_consider[best_idx]
            best_feature_name = all_features[best_feature_idx]
            best_feature_score = scores[best_idx]

            features_to_consider.remove(best_feature_idx)
            if direction == 'forward':
                selected_features.append(best_feature_idx)
                print("Adding feature {}".format(best_feature_name))
            elif direction == 'backward':
                selected_features.remove(best_feature_idx)

            selected_features_names = [all_features[feature_idx] for feature_idx in selected_features]
            history.append({'score': best_feature_score, 'selected_features': selected_features_names})

            if nb_features == "auto" and best_feature_score > autostop_maximum:
                autostop_maximum = best_feature_score
                autostop_counter = 0
            else:
                autostop_counter += 1
                if autostop_counter == autostop_patience:
                    break


        X_future.release()
        y_future.release()
        self.select_properties(selected_features_names)
        return history
            