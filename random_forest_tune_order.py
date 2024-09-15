import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json
from tqdm import tqdm
import logging
from collections import defaultdict


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


def separate_data(data, set_length):
    return [value for value in data.values() if len(value) == set_length]


def create_incorrect_order(tunes):
    if len(tunes) == 2:
        return tunes[::-1]  # Reverse the order for sets of size 2
    elif len(tunes) == 3:
        return [tunes[1], tunes[2], tunes[0]]  # Rearrange for sets of size 3
    else:
        raise ValueError(f"Unexpected set size: {len(tunes)}")


def prepare_data(set_collection, property_indices=None):
    X = []
    y = []

    for tunes in set_collection:
        # Correct order
        tunes_correct = sorted(tunes, key=lambda x: x['position'])
        set_features_correct = []
        for tune in tunes_correct:
            features = [tune['type'], tune['meter'], tune['mode'], tune['tonic']]
            if property_indices is not None:
                features = [features[i] for i in property_indices]
            set_features_correct.extend(features)

        X.append(set_features_correct)
        y.append(1)  # Correct order

        # Incorrect order
        tunes_incorrect = create_incorrect_order(tunes_correct)
        set_features_incorrect = []
        for tune in tunes_incorrect:
            features = [tune['type'], tune['meter'], tune['mode'], tune['tonic']]
            if property_indices is not None:
                features = [features[i] for i in property_indices]
            set_features_incorrect.extend(features)

        X.append(set_features_incorrect)
        y.append(0)  # Incorrect order

    return np.array(X), np.array(y)


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    support = len(y_true)
    return precision, recall, f1, support, accuracy


def random_forest_tune_order(X, y, feature_names):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Initialize LabelEncoder for each feature
    label_encoders = [LabelEncoder() for _ in range(X.shape[1])]

    # Fit and transform each feature
    X_encoded = np.array([le.fit_transform(X[:, i]) for i, le in enumerate(label_encoders)]).T

    fold_metrics = defaultdict(list)
    for fold, (train_index, test_index) in enumerate(skf.split(X_encoded, y), 1):
        X_train, X_test = X_encoded[train_index], X_encoded[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['f1'].append(f1)
        fold_metrics['accuracy'].append(accuracy)

        logging.info(
            f"Fold {fold} results: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, accuracy={accuracy:.4f}, support={len(y_test)}")
        logging.info(f"y_test unique values: {np.unique(y_test, return_counts=True)}")
        logging.info(f"y_pred unique values: {np.unique(y_pred, return_counts=True)}")

    # Calculate aggregate statistics
    aggregate_results = {}
    for metric, values in fold_metrics.items():
        aggregate_results[metric] = {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values))
        }

    # Calculate feature importance
    clf.fit(X_encoded, y)  # Fit on entire encoded dataset for overall feature importance

    # Aggregate feature importances
    feature_importance_dict = defaultdict(float)
    for feature, importance in zip(feature_names, clf.feature_importances_):
        feature_type = feature.split('_')[0]  # Extract the feature type (e.g., 'tonic' from 'tonic_1')
        feature_importance_dict[feature_type] += importance

    # Convert to list and sort
    feature_importance = [
        {'feature': feature, 'importance': importance}
        for feature, importance in feature_importance_dict.items()
    ]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)

    return {
        'fold_results': aggregate_results,
        'feature_importance': feature_importance
    }


def analyze_folds(fold_results):
    metrics = ['precision', 'recall', 'f1', 'support', 'accuracy']
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


def analyze_feature_importances(feature_importances):
    all_features = set(feature_importances[0].keys())  # Assuming all folds have the same features
    aggregated_importances = {}

    for feature in all_features:
        values = [fold_imp[feature] for fold_imp in feature_importances]
        aggregated_importances[feature] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

    # Sort features by mean importance
    sorted_features = sorted(aggregated_importances.items(), key=lambda x: x[1]['mean'], reverse=True)
    return dict(sorted_features)


def main():
    set_collection = get_data()
    results = {}

    properties = ['all', 'type', 'meter', 'mode', 'tonic']
    property_indices = {
        'all': None,
        'type': [0],
        'meter': [1],
        'mode': [2],
        'tonic': [3]
    }

    for set_length in tqdm([2, 3]):
        split_collection = separate_data(set_collection, set_length)
        results[f"set_size_{set_length}"] = {}

        for prop in properties:
            X, y = prepare_data(split_collection, property_indices[prop])

            # Define feature names based on the property and set length
            if prop == 'all':
                feature_names = [f"{p}_{i}" for i in range(1, set_length + 1) for p in
                                 ['type', 'meter', 'mode', 'tonic']]
            else:
                feature_names = [f"{prop}_{i}" for i in range(1, set_length + 1)]

            logging.info(f"Set size {set_length}, Property {prop}: X shape = {X.shape}, y shape = {y.shape}")
            logging.info(
                f"Set size {set_length}, Property {prop}: Unique y values = {np.unique(y, return_counts=True)}")

            classifier_results = random_forest_tune_order(X, y, feature_names)

            results[f"set_size_{set_length}"][prop] = classifier_results

    # Save results to JSON
    with open("results/set_order_classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to results/set_order_classifier_results.json")


if __name__ == "__main__":
    main()