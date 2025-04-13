import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

df = pd.read_csv('exam_2_data.csv')
df_cleaned = df.dropna(subset=["Excess Readmission Ratio", "Measure Name"]).copy()
le = LabelEncoder()
df_cleaned["Measure Code"] = le.fit_transform(df_cleaned["Measure Name"])
X = df_cleaned[["Excess Readmission Ratio"]].values
y = df_cleaned["Measure Code"].values

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
        return {"feature_index": best_split["feature_index"],
                "threshold": best_split["threshold"],
                "left": left_tree,
                "right": right_tree}

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
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForest(n_estimators=5, max_depth=4)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
