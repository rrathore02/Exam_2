import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        data = np.column_stack((X, y))
        self.tree = self._build_tree(data)

    def _build_tree(self, data, depth=0):
        X, y = data[:, :-1], data[:, -1]
        if len(set(y)) == 1 or len(data) < self.min_samples_split or depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        best_split = self._find_best_split(data)
        if not best_split:
            return Counter(y).most_common(1)[0][0]

        left_tree = self._build_tree(best_split["left"], depth + 1)
        right_tree = self._build_tree(best_split["right"], depth + 1)
        return {
            "feature_index": best_split["feature_index"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree
        }

    def _find_best_split(self, data):
        best_gini = float("inf")
        best_split = None
        n_features = data.shape[1] - 1

        for feature_index in range(n_features):
            thresholds = np.unique(data[:, feature_index])
            for threshold in thresholds:
                left = data[data[:, feature_index] <= threshold]
                right = data[data[:, feature_index] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = self._gini_index([left[:, -1], right[:, -1]])
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left": left,
                        "right": right
                    }
        return best_split

    def _gini_index(self, groups):
        n_instances = sum(len(group) for group in groups)
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = sum((np.sum(group == cls) / size) ** 2 for cls in np.unique(group))
            gini += (1 - score) * (size / n_instances)
        return gini

    def _predict_one(self, inputs, node):
        if not isinstance(node, dict):
            return node
        if inputs[node["feature_index"]] <= node["threshold"]:
            return self._predict_one(inputs, node["left"])
        else:
            return self._predict_one(inputs, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, n_estimators=5, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
    
##    I used generative ai for some parts
