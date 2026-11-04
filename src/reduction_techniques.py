"""FCNN, ENNTh and DROP4 instance reduction methods."""

import numpy as np
from knn_classifier import KNNClassifier


def fcnn1_rule(X, y, verbose=True, **kwargs):
    """
    Computes the Fast Condensed Nearest Neighbor (FCNN1) subset.

    Parameters:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
    y (numpy.ndarray): Label array of shape (n_samples,)

    Returns:
    S_indices (list): Indices of the selected subset in the original training set
    X_S (numpy.ndarray): Condensed feature matrix S (training-set consistent subset of X)
    y_S (numpy.ndarray): Condensed label array

    Based on Angiulli, F. (2007). Fast Nearest Neighbor Condensation for Large Data
    Sets Classification. IEEE Transactions On Knowledge And Data Engineering, 19(11),
    1450-1464. https://doi.org/10.1109/tkde.2007.190645

    Voronoi enemies of a point p: set of points that are inside the Voronoi cell of
    p but are misclassified by the subset.
    """

    classes = np.unique(y) # obtain the classes from the training dataset X
    S_indices = []

    if "k" in kwargs.keys():
        k = kwargs["k"]
        kwargs.pop("k")
    else:
        k = 1

    # Initialize subset S with seeds as the centroids of each class
    for c in classes:
        X_c = X[y == c] # obtain the instances of X that are labeled as c
        centroid = X_c.mean(axis=0) # compute a mean or centroid attribute vector for all instances

        # Find the actual instance closest to the centroid
        dist_to_centroid = np.linalg.norm(X_c - centroid, axis=1) # compute the distance between the instances and their centroid
        closest_idx_in_Xc = np.argmin(dist_to_centroid)

        # Map back to the absolute index in X
        abs_idx = np.where(y == c)[0][closest_idx_in_Xc]
        S_indices.append(abs_idx) # save the idx of the instance in X rather than the instance itself for efficiency

    S_indices = list(set(S_indices)) # ensure uniqueness

    iteration = 1
    while True:
        delta_S = []

        # Current condensed subset S
        X_S = X[S_indices]
        y_S = y[S_indices]

        # Fit k-NN on the current subset S
        nn = KNNClassifier(
            k=min(k, len(S_indices)), **kwargs
        ).fit(X_S, y_S) # y_S unused

        # Find Voronoi enemies of S: instances in X that are closest to another instance in S with different label
        # Find the nearest instance in S for every instance in X
        indices_in_S = np.empty(X.shape[0], dtype=int) # maps each instance in X to its closest prototype in S
        predictions = np.empty(X.shape[0], dtype=y.dtype)
        distances_X_to_S = np.empty((X.shape[0], len(S_indices)), dtype=float) # (n_samples, X_S samples)
        for i, x_i in enumerate(X):
            # Get distances to find the absolute closest prototype for Voronoi cell assignment
            distances = nn.compute_distances(query_point=x_i) # distances between x_i and all the prototype instances inside S
            distances_X_to_S[i, :] = distances

            # Prototype in S that is closest to x_i, relative idx in S_indices
            indices_in_S[i] = np.argmin(distances)

            # Get the k-NN majority vote prediction
            # We already computed these distances. predict() would compute them
            # a second time. Use the same neighbour and voting rules directly.
            count = min(nn.k, len(distances))
            neighbours = (np.argsort(distances) if count == len(distances)
                          else np.argpartition(distances, count)[:count])
            predictions[i] = nn._get_vote(distances[neighbours], y_S[neighbours])

        # For each prototype point in S, find its Voronoi enemies
        for i, p_idx in enumerate(S_indices):
            voronoi_cell_mask = (indices_in_S == i) # mask that points which instances of X correspond to the prototype p_idx in S

            # we want the instances that correspond to the Voronoi cell of p and that are misclassified by the knn
            # this is, the instances that using S for prediction in knn were wrong
            enemy_mask = voronoi_cell_mask & (predictions != y) # Voronoi enemies
            enemy_mask[p_idx] = False # exclude the prototype itself from being its own enem
            enemy_indices = np.where(enemy_mask)[0] # returns the idx of the enemies in X

            if len(enemy_indices) > 0: # if the prototype p_idx has at least one enemy in its territory
                # FCNN1 Representative: The enemy nearest to the prototype p

                # Calculate distance from the prototype to every one of its Voronoi enemies
                # dist_to_enemies = nn.compute_distances(query_point=X[p_idx], data=X[enemy_indices])
                dist_to_enemies = distances_X_to_S[enemy_indices, i] # faster version than the one above

                # Select the nearest enemy
                nearest_enemy_idx = enemy_indices[np.argmin(dist_to_enemies)] # extract idx in X of that specific enemy
                delta_S.append(nearest_enemy_idx)
                # FCNN hunts for boundaries where the classes clash.
                # When a prototype finds an enemy in its territory, FCNN promotes the closest enemy to become a new
                # prototype in S.

        # Duplicate feature vectors can nominate a prototype already in S.
        # Stop if this pass adds nothing, rather than repeating forever.
        delta_S = [index for index in delta_S if index not in S_indices]
        if not delta_S:
            if verbose:
                print(f"FCCN1 converged in {iteration} iterations.")
            break

        # Update S with the new representatives
        S_indices.extend(delta_S)
        S_indices = list(set(S_indices))
        iteration += 1

    return S_indices, X[S_indices], y[S_indices]

ENNTH_MU = 0.8 # hardcoding this prob threshold as the paper showed this yielded best results between 0.6, 0.7, 0.8


def ennth(X_train: np.ndarray, y_train: np.ndarray, k: int) -> (np.ndarray, np.ndarray):

    mu = ENNTH_MU

    # some protection against edge cases
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError('X_train and y_train must have the same size')
    if not 1 <= k <= X_train.shape[0] - 1:
        raise ValueError('k must be between 1 and the number of other training cases')
    if not 0 <= mu <= 1:
        raise ValueError('mu must be between 0 and 1')

    misclassified_indices = []

    # single path as this is a batched reduction technique
    for i in range(X_train.shape[0]):
        # Leave out the query itself. Only copy a distance vector, not the whole training matrix for every query.
        # The neighbour order is unchanged.
        dists = np.delete(inverse_distance_ennth(X_train[i], X_train), i)
        top_k = np.argpartition(dists, -k)[-k:] # want the top k inverse distances' indices
        class_probabilities = {}
        for j in top_k:
            # Removing position i shifts later positions by one. This adjustment is important to make sure labels stay aligned
            label = y_train[j + int(j >= i)]
            # get class distribution among neighbours to reach probabilities
            if label not in class_probabilities:
                class_probabilities[label] = 0
            class_probabilities[label] += dists[j]
        total_for_all_classes = sum(class_probabilities.values())
        for key in class_probabilities:
            class_probabilities[key] /= total_for_all_classes

        # check if prediction based on most likely class is wrong or not confident enough
        prediction = max(class_probabilities, key=class_probabilities.get)
        if prediction != y_train[i] or class_probabilities[prediction] < mu:
            # discard ith index of training set if classification is wrong or probability below threshold
            misclassified_indices.append(i)

    # An empty list must still become an integer array for np.delete().
    misclassified_indices = np.asarray(misclassified_indices, dtype=int)

    # batch remove all ineligible points after full iteration
    X_train = np.delete(X_train, misclassified_indices, axis=0)
    y_train = np.delete(y_train, misclassified_indices, axis=0)
    return X_train, y_train

def inverse_distance_ennth(query: np.ndarray, rest_of_set: np.ndarray) -> np.ndarray:
    return 1/(1+euclidean(query, rest_of_set))

def euclidean(query: np.ndarray, rest_of_set: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((rest_of_set - query)**2, axis=1))

def drop4(X_train: np.ndarray, y_train: np.ndarray, k: int, debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Incremental implementation of DROP4 - faster than previous attempts"""
    X_original = np.asarray(X_train)
    X = np.asarray(X_train, dtype=float) # float across the board as our current datasets only contain these
    y = np.asarray(y_train)

    # protect against bad input
    if X.ndim != 2:
        raise ValueError("X_train must be a 2D array.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y_train must be 1D and match X_train.")

    n = len(X)

    # Pairwise squared Euclidean distances. This helps us keep track of neighbour candidates as queries are inactivated.
    distances = np.empty((n, n), dtype=float)
    for i in range(n):
        delta = X - X[i]
        distances[i] = np.sum(delta * delta, axis=1)

    # Precompute each query's ranking, explicitly excluding itself.
    # This handles duplicate observations where self may not sort first.
    ranked_neighbours = []
    for i in range(n):
        order = np.argsort(distances[i], kind="stable")
        # Indices fit in int32 for our datasets. This halves the memory used
        # by the rankings without changing distances or neighbour ordering.
        ranked_neighbours.append(order[order != i].astype(np.int32))

    active = np.ones(n, dtype=bool)
    active_count = n # use this to avoid taking length of active each time

    # Maintain lists for all original queries, even removed queries.
    # This bit is important because DROP4 inherits from DROP2, which says we measure the impact of a query's removal
    # on the ORIGINAL dataset, rather than just what's left
    neighbour_lists = [
        row[: k + 1].astype(int).tolist()
        for row in ranked_neighbours
    ]

    # Cursor pointing to the next ranked candidate to examine for replacement.
    next_rank = np.full(n, k + 1, dtype=np.intp)

    # holders[p]: queries whose stored k+1 list contains prototype p.
    holders = [set() for _ in range(n)]

    # associates[p]: queries whose first k neighbours include prototype p.
    # Queries may themselves have already been removed from the active set - again needed by DROP2
    associates = [set() for _ in range(n)]

    for query, neighbours in enumerate(neighbour_lists):
        for position, prototype in enumerate(neighbours):
            holders[prototype].add(query)
            if position < k:
                associates[prototype].add(query)

    def majority_label(neighbour_indices):
        # gets most common label among neighbours for classification
        labels, counts = np.unique(
            y[np.asarray(neighbour_indices, dtype=np.intp)],
            return_counts=True,
        )
        # Deterministic tie-break: first label returned by np.unique.
        return labels[np.argmax(counts)]

    def predict(query):
        neighbours = neighbour_lists[query][:k]
        # hopefully shouldn't happen on a decent-sized set but gotta protect against it
        if len(neighbours) < k:
            raise RuntimeError("Fewer than k active neighbours are available.")
        return majority_label(neighbours)

    def can_remove(prototype: int) -> bool:
        # Check whether removal preserves associate classification accuracy.
        correct_with = 0
        correct_without = 0

        # Only associates can have their predictions changed by removing p.
        # Removed queries remain included in this score.
        for query in associates[prototype]:
            neighbours = neighbour_lists[query]

            prediction_with = majority_label(neighbours[:k])

            # This query has p among its first k neighbours. Its stored k+1
            # list supplies the replacement when p is hypothetically removed.
            neighbours_without = [
                q for q in neighbours if q != prototype
            ][:k]

            prediction_without = majority_label(neighbours_without)

            correct_with += prediction_with == y[query]
            correct_without += prediction_without == y[query]

        return correct_without >= correct_with

    def next_active_neighbour(query: int) -> int | None:
        # Find the next active prototype in this query's precomputed ranking.
        rank = int(next_rank[query])
        ranking = ranked_neighbours[query]
        current = set(neighbour_lists[query])

        # scan the sorted list of other neighbours by proximity until we find one that hasn't been deactivated yet
        while rank < len(ranking):
            candidate = int(ranking[rank])
            rank += 1

            if active[candidate] and candidate not in current:
                next_rank[query] = rank
                return candidate

        next_rank[query] = rank
        return None

    def remove_prototype(prototype: int) -> None:
        # Remove p and repair only neighbour lists that currently contain it.
        nonlocal active_count # prevent clash with global scope

        affected_queries = list(holders[prototype])

        active[prototype] = False
        active_count -= 1

        for query in affected_queries:
            neighbours = neighbour_lists[query]
            position = neighbours.index(prototype)

            # If p was in the top k, the old (k+1)th neighbour is promoted
            # into the top k and becomes a new associate of this query.
            if position < k:
                associates[prototype].discard(query)

                if len(neighbours) > k:
                    promoted = neighbours[k]
                    associates[promoted].add(query)

            neighbours.remove(prototype)
            holders[prototype].discard(query)

            # An active query excludes itself from its neighbour list.
            # Near the stopping boundary, it may have only k other active
            # prototypes, so a k+1 buffer is no longer always possible.
            target_length = min(
                k + 1,
                active_count - int(active[query]),
            )

            if len(neighbours) < target_length:
                replacement = next_active_neighbour(query)

                if replacement is None:
                    raise RuntimeError(
                        "Could not find a replacement neighbour."
                    )

                neighbours.append(replacement)
                holders[replacement].add(query)
                # The new entry is the replacement buffer at position k.

        # we don't need to worry about what other queries have this prototype as a k or k+1 neighbour, so we can
        # clean up these rows of holders and associates
        holders[prototype].clear()
        associates[prototype].clear()

    # first pass to carefully remove noise - DROP3 uses ENN but DROP4 also has to consider whether removing an index
    # would harm classification of associates
    for i in range(n):
        if active_count <= k + 1:
            break

        if predict(i) != y[i] and can_remove(i):
            remove_prototype(i)

            if debug:
                print(
                    f"Noise pass: removed {i}; "
                    f"{active_count} prototypes remain"
                )

    # Order remaining prototypes furthest from their nearest enemy first.
    remaining = np.flatnonzero(active)
    enemy_distances = np.full(n, np.inf, dtype=float)

    for i in remaining:
        enemies = remaining[y[remaining] != y[i]]

        if len(enemies):
            enemy_distances[i] = np.min(distances[i, enemies])
        # If no active enemy exists, infinity places this instance first, but we probably won't need to worry about it

    order = remaining[
        np.argsort(-enemy_distances[remaining], kind="stable")
    ]

    # Second and final pass uses the ordered list to incrementally reduce, this time only focussing on
    # whether the removal of a query would harm classification of its associates overall
    for prototype in order:
        if active_count <= k + 1:
            break

        # Defensive check; order contains unique indices.
        if not active[prototype]:
            continue

        if can_remove(int(prototype)):
            remove_prototype(int(prototype))

            if debug:
                print(
                    f"Reduction pass: removed {prototype}; "
                    f"{active_count} prototypes remain"
                )

    return X_original[active], y[active]
