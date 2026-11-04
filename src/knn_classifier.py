"""kNN classifier, shared by the harness and reduction techniques."""

import numpy as np
import pandas as pd

class KNNClassifier:
    def __init__(self,
                 k: int,
                 distance_metric: str = "euclidean",
                 voting_policy: str = "majority",
                 retention_policy: str = "Never",
                 ):

        if retention_policy not in ("Never", "Always", "DC", "OG"): # DC: Different Class retention, OG: Oblivion by Goodness
            raise ValueError('retention_policy must be "Never", "Always", "DC" or "OG')

        self.k = k
        self.distance_metric = distance_metric
        self.voting_policy = voting_policy
        self.retention_policy = retention_policy

        # Datasets are kept as attributes
        self.X_train = None
        self.y_train = None

        # Goodness (OB retain policy)
        self.goodness = None
        self.goodness_0 = 0.5
        self.learning_rate = 0.2

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Store the training datasets.
        X_train and y_train can be assumed to be normalised and without missing data, due to pre-processing"""

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of training instances must equal number of training classifications")

        self.X_train = np.asarray(X_train, dtype=float, copy=True) # copy as True to avoid modifying the original data during retention policies
        self.y_train = np.asarray(y_train, copy=True)

        if self.retention_policy == "OG":
            self.goodness = np.full(self.X_train.shape[0], self.goodness_0) # initialize goodness for every training instance

        if self.distance_metric == "ivdm":
            self._init_ivdm_structure(X_train, y_train)
            self.refit_ivdm()

        return self

    def _init_ivdm_structure(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        self.n_classes = self.classes_.size
        self.n_bins = max(5, self.n_classes)  # heuristic mentioned by Wilson & Martinez on p14

        self.feat_min_ = X_train.min(axis=0)
        self.feat_max_ = X_train.max(axis=0)
        self.bin_width_ = (self.feat_max_ - self.feat_min_) / self.n_bins
        self.constant_ = self.bin_width_ == 0  # constant features contribute 0 distance -- just a precaution

    def refit_ivdm(self):
        n_train_rows, n_train_features = self.X_train.shape
        y_encoded = np.searchsorted(self.classes_, self.y_train)

        # priors for classes are proportions of training instances in each class
        self.class_priors_ = np.bincount(y_encoded, minlength=self.n_classes) / n_train_rows

        self.bin_counts_ = np.zeros((n_train_features, self.n_bins, self.n_classes))
        self.bin_probs_ = np.tile(self.class_priors_, (n_train_features, self.n_bins, 1))
        self.bin_midpoints_ = np.zeros((n_train_features, self.n_bins))

        for a in range(n_train_features):
            if self.constant_[a]:
                continue
            idx = self.ivdm_bin_index(self.X_train[:, a], a)
            counts = np.bincount(idx * self.n_classes + y_encoded, minlength=self.n_bins * self.n_classes).reshape(self.n_bins, self.n_classes)
            denom = counts.sum(axis=1, keepdims=True)
            safe = np.where(denom > 0, denom, 1.0)
            probs = np.where(denom > 0, counts / safe, self.class_priors_)  # empty bins -> priors

            self.bin_counts_[a] = counts
            self.bin_probs_[a] = probs
            self.bin_midpoints_[a] = (
                    self.feat_min_[a] + self.bin_width_[a] * (np.arange(self.n_bins) + 0.5)
            )

        self.ivdm_values = self.compute_ivdm_values(self.X_train)

    def ivdm_bin_index(self, col, a):
        # figure out what discretized bin each value of a feature belongs to, based on the min/max and bin width learned during fit
        idx = np.floor((col - self.feat_min_[a]) / self.bin_width_[a]).astype(int)
        return np.clip(idx, 0, self.n_bins - 1)  # max value and out-of-range values

    def compute_ivdm_values(self, X_train):
        # give P(c | x_a) for every value of every feature of every instance in X, using the bin midpoints and probabilities learned during fit
        X_train = np.asarray(X_train, dtype=float)
        out = np.zeros((X_train.shape[0], X_train.shape[1], self.n_classes))
        for a in range(X_train.shape[1]):
            if self.constant_[a]:
                continue
            for c in range(self.n_classes):
                # for this feature, for this class, interpolate the probability for each instance in X using the bin midpoints and probabilities learned during fit
                out[:, a, c] = np.interp(
                    X_train[:, a], self.bin_midpoints_[a], self.bin_probs_[a, :, c]
                )
        return out

    def _ivdm_distance(self, query_point: np.ndarray) -> np.ndarray:
        # first figure out conditional probabilities per class for each feature value of query point
        ivdm_data_for_query = self.compute_ivdm_values(np.atleast_2d(query_point))[0]

        diff = np.abs(self.ivdm_values - ivdm_data_for_query)
        per_feature = diff.sum(axis=2)  # sum over classes to get a single distance per feature
        return np.sqrt((per_feature ** 2).sum(axis=1))

    def _euclidean_distance(self, train_points: np.ndarray, query_point: np.ndarray) -> np.ndarray:
        """Return the Euclidean distance to each training point."""
        return np.sqrt(((train_points - query_point) ** 2).sum(axis=1))

    def _manhattan_distance(self, train_points: np.ndarray, query_point: np.ndarray) -> float:
        """
        Manhattan distance between all points in training set and query point.
        """
        return (np.abs(train_points - query_point).sum(1))

    def _canberra_distance(self, train_points: np.ndarray, query_point: np.ndarray) -> np.ndarray:
        """Return the Canberra distance to each training point."""
        numerator = np.abs(train_points - query_point)
        denominator = np.abs(train_points) + np.abs(query_point)
        # Two zeros add zero to the distance.
        parts = np.divide(numerator, denominator,
                          out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
        return parts.sum(axis=1)

    def compute_distances(self, query_point: np.ndarray, data=None) -> np.ndarray:
        """Compute distances based on the chosen metric."""
        if data is None:
            data = self.X_train

        if self.distance_metric == "euclidean":
            return self._euclidean_distance(data, query_point)
        elif self.distance_metric == "manhattan":
            return self._manhattan_distance(data, query_point)
        elif self.distance_metric == "canberra":
            return self._canberra_distance(data, query_point)
        elif self.distance_metric == "ivdm":
            return self._ivdm_distance(query_point)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _get_vote(self, distances_array: np.ndarray, labels: np.ndarray):
        """Apply the voting policy."""
        if self.voting_policy == "majority":
            # Select the modal label category from k nearest neighbours
            vals, counts = np.unique(labels, return_counts=True)
            tied_labels = vals[counts == counts.max()]
            if tied_labels.size == 1:
                return tied_labels[0]

            # Among labels tied on vote count, choose the one with the
            # closest neighbour. np.unique keeps labels sorted, so argmin
            # also gives the requested sorted-label fallback on equal distances.
            closest_distances = np.array([
                distances_array[labels == label].min() for label in tied_labels
            ])
            return tied_labels[closest_distances.argmin()]

        elif self.voting_policy in ("inverse", "inverse_distance"):
            # Select category based on inverse distance weighting
            inverse_distances = np.divide( # avoid division by zero in distances_array
                1.0, distances_array,
                out=np.full_like(distances_array, np.inf, dtype=float), # when distance is zero we assign np.inf as 1/dist
                where=distances_array != 0
            )
            # get the unique labels and their corresponding ints (one per distinct label) so we can use np.bincount
            # to sum the total inverse distances for each label
            values, label_integers = np.unique(labels, return_inverse=True)
            counts = np.bincount(label_integers, weights=inverse_distances)
            tied_labels = values[counts == counts.max()]
            if tied_labels.size == 1:
                return tied_labels[0]

            # If inverse-distance scores tie, prefer the label appearing
            # most often among these k neighbours. Since values are sorted,
            # argmax resolves any remaining tie to the first sorted label.
            tied_counts = np.array([
                np.count_nonzero(labels == label) for label in tied_labels
            ])
            return tied_labels[tied_counts.argmax()]

        raise ValueError(f"Unknown voting policy: {self.voting_policy}")

    def _retain_policy(self, idx: np.ndarray, success: bool, x: np.ndarray, true_label):
        """ Apply the retention policy.
        idx: indexes from self.X_train corresponding to the k nearest neighbors.
        """
        if self.retention_policy == "Always" or (self.retention_policy == "DC" and not success):
            # x: new instance to add
            self.X_train = np.vstack((self.X_train, x))
            self.y_train = np.append(self.y_train, true_label)
            if self.distance_metric == "ivdm":
                self.ivdm_values = np.vstack((self.ivdm_values, self.compute_ivdm_values(np.atleast_2d(x))))

        elif self.retention_policy == "OG":
            # k nearest neighbours update their goodness
            binary_reward = np.asarray(self.y_train[idx] == true_label, dtype=float) # 1 or 0
            self.goodness[idx] = self.goodness[idx] + self.learning_rate * (binary_reward - self.goodness[idx])

            # forgetting strategy: remove cases that consistenly demonstrate poor classification behavior
            idx_to_be_removed = idx[self.goodness[idx] < self.goodness_0]
            if len(idx_to_be_removed) > 0:
                self.X_train = np.delete(self.X_train, idx_to_be_removed, axis=0)
                self.y_train = np.delete(self.y_train, idx_to_be_removed, axis=0)
                self.goodness = np.delete(self.goodness, idx_to_be_removed, axis=0)
                if self.distance_metric == "ivdm":
                    self.ivdm_values = np.delete(self.ivdm_values, idx_to_be_removed, axis=0)
        else:
            pass

    def predict(self, x_test: np.ndarray):

        distances = self.compute_distances(query_point=x_test)

        # Reduction or OG may leave fewer than k cases. Use the cases still available.
        if len(distances) == 0:
            raise ValueError("The kNN case base is empty.")
        k_available = min(self.k, len(distances))
        if k_available == len(distances):
            idx = np.argsort(distances)
        else:
            # Keep the team's neighbour selection when at least k+1 cases remain.
            idx = np.argpartition(distances, self.k)[:self.k]

        prediction = self._get_vote(distances[idx], self.y_train[idx]) # distances of the k nearest instances, labels of the k nearest instances

        return prediction, idx

    def evaluate_instance(self, x_test: np.ndarray, y_true) -> bool:
        """Predict a single instance and apply the retention policy.
        X_test: (num_features,)
        y_true: label
        """

        prediction, idx = self.predict(x_test) # idx = k nearest neighbours
        success = bool(prediction == y_true) # prediction = winning_label

        # Apply retention policy
        self._retain_policy(idx, success, x_test, y_true)

        return success

    def evaluate_dataset(self, test_df: pd.DataFrame) -> np.ndarray:
        """Evaluate an entire dataset dataframe and print the summary statistics."""

        # Split test data into features and labels
        X_test = test_df.iloc[:, :-1].to_numpy(dtype=float)
        y_test = test_df.iloc[:, -1].to_numpy() # extract the last column, which are the labels for all instances

        successes = []
        for x_t, y_t in zip(X_test, y_test):
            success = self.evaluate_instance(x_t, y_t)
            successes.append(success)

        accuracy = np.mean(successes) * 100
        print(f"Percentage success for knn with k:{self.k}, "
              f"distance metric: {self.distance_metric}, "
              f"voting policy: {self.voting_policy}, "
              f"retention: {self.retention_policy}: {accuracy:.3f}%")
        print(f"Size of train set at end of evaluation: {self.X_train.shape[0]}")

        return np.asarray(successes, dtype=bool)
