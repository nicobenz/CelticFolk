from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_data():
    with open("data/sets.json") as f:
        data = json.load(f)

    set_collection = {}
    for item in data:
        tune = {
            "position": item["settingorder"],
            "type": item["type"],
            "meter": item["meter"],
            "mode": item["mode"][1:],
            "tonic": item["mode"][:1],
        }

        if item["tuneset"] not in set_collection:
            set_collection[item["tuneset"]] = [tune]
        else:
            set_collection[item["tuneset"]].append(tune)
    return set_collection


def split_data(set_collection, features, set_size):
    tune_set = []
    for k, v in set_collection.items():
        if len(v) == set_size:
            tune_set.extend(v)

    df = pd.DataFrame(tune_set).copy()  # Create an explicit copy

    if features == ["all"]:
        X = df.drop("position", axis=1)
    else:
        X = df[features].copy()  # Create a copy of the selected features
    y = df["position"]

    # Encode categorical variables
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X.loc[:, col] = le.fit_transform(X[col])

    return X, y


def random_forest_tune_position(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    fold_results = []
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        fold_results.append({
            'fold': fold,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'support': int(len(y_test))
        })

    # Calculate feature importance
    clf.fit(X, y)  # Fit on entire dataset for overall feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'fold_results': fold_results,
        'feature_importance': feature_importance.to_dict(orient='records')
    }


def analyze_folds(fold_results):
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'support']
    analysis = {}

    for metric in metrics:
        values = [fold[metric] for fold in fold_results]
        analysis[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

    return analysis

def main():
    data = get_data()
    set_sizes = [2, 3]
    feature_combinations = [
        ["all"],
        ["type"],
        ["meter"],
        ["mode"],
        ["tonic"],
        ["type", "meter"],
        ["type", "mode"],
        ["type", "tonic"],
        ["meter", "mode"],
        ["meter", "tonic"],
        ["mode", "tonic"]
    ]

    results = {}

    for set_size in set_sizes:
        results[f"set_size_{set_size}"] = {}
        for features in tqdm(feature_combinations):
            X, y = split_data(data, features, set_size)
            feature_key = "_".join(features)
            rf_results = random_forest_tune_position(X, y)
            results[f"set_size_{set_size}"][feature_key] = {
                'metrics': analyze_folds(rf_results['fold_results']),
                'feature_importance': rf_results['feature_importance']
            }

    # Save results to a JSON file
    with open("results/random_forest_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()