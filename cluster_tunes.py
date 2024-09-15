import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from kneed import KneeLocator
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def prepare_data():
    with open("data/sets.json") as f:
        data = json.load(f)

    tune_sets = defaultdict(list)
    for item in data:
        tune = {
            "id": item["tune_id"],
            "type": item["type"],
            "meter": item["meter"],
            "mode": item["mode"][1:],
            "tonic": item["mode"][:1],
            "tuneset": item["tuneset"],
            "name": item["name"],
            "setting_id": item["setting_id"]
        }
        tune_sets[item["tuneset"]].append(tune)

    sets_of_two = [s for s in tune_sets.values() if len(s) == 2]
    sets_of_three = [s for s in tune_sets.values() if len(s) == 3]

    return sets_of_two, sets_of_three


def encode_features(tune_sets):
    all_tunes = [tune for tune_set in tune_sets for tune in tune_set]
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform([(t['type'], t['meter'], t['mode'], t['tonic']) for t in all_tunes])
    return encoded_features, encoder


def elbow_method(data, max_clusters=100):
    inertias = []
    for k in tqdm(range(1, max_clusters + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(range(1, max_clusters + 1), inertias, curve="convex", direction="decreasing")
    elbow_point = kl.elbow

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=list(range(1, max_clusters + 1)), y=inertias, mode='lines+markers', name='Inertia'))
    if elbow_point:
        fig.add_vline(x=elbow_point, line_dash="dash", line_color="red",
                      annotation_text=f"Elbow point: {elbow_point}",
                      annotation_position="top right")
    fig.update_layout(title='Elbow Method for Optimal k', xaxis_title='Number of clusters (k)',
                      yaxis_title='Inertia', showlegend=True)

    return elbow_point, inertias, fig


def run_k_means(data, tune_sets, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    set_clusters = []
    for i in range(0, len(cluster_labels), len(tune_sets[0])):
        set_clusters.append(cluster_labels[i:i + len(tune_sets[0])])

    total_sets = len(set_clusters)

    if len(tune_sets[0]) == 2:
        same_cluster_count = sum(len(set(clusters)) == 1 for clusters in set_clusters)
        same_cluster_percentage = (same_cluster_count / total_sets * 100) if total_sets > 0 else 0

        result = {
            "total_sets": total_sets,
            "same_cluster": same_cluster_count,
            "same_cluster_percentage": same_cluster_percentage,
        }

    elif len(tune_sets[0]) == 3:
        all_same_cluster_count = sum(len(set(clusters)) == 1 for clusters in set_clusters)
        two_same_cluster_count = sum(len(set(clusters)) == 2 for clusters in set_clusters)

        all_same_cluster_percentage = (all_same_cluster_count / total_sets * 100) if total_sets > 0 else 0
        two_same_cluster_percentage = (two_same_cluster_count / total_sets * 100) if total_sets > 0 else 0

        result = {
            "total_sets": total_sets,
            "all_same_cluster": all_same_cluster_count,
            "all_same_cluster_percentage": all_same_cluster_percentage,
            "two_same_cluster": two_same_cluster_count,
            "two_same_cluster_percentage": two_same_cluster_percentage,
        }

    cluster_sizes = Counter(cluster_labels)
    result["cluster_distribution"] = dict(cluster_sizes)

    return convert_to_serializable(result)

def main():
    sets_of_two, sets_of_three = prepare_data()

    results = {}
    for set_size, tune_sets in [("two", sets_of_two), ("three", sets_of_three)]:
        print(f"\nAnalyzing sets of {set_size} tunes:")
        encoded_features, encoder = encode_features(tune_sets)
        elbow_point, inertias, fig = elbow_method(encoded_features)
        print(f"Optimal number of clusters for sets of {set_size}: {elbow_point}")

        # Save elbow plot data
        elbow_data = {
            'elbow_point': int(elbow_point) if elbow_point is not None else None,
            'inertias': [float(i) for i in inertias],
            'max_clusters': len(inertias)
        }
        with open(f'elbow_plot_data_{set_size}.json', 'w') as f:
            json.dump(elbow_data, f)

        fig.write_html(f"elbow_plot_sets_of_{set_size}.html")

        analysis = run_k_means(encoded_features, tune_sets, elbow_point)
        results[f"sets_of_{set_size}"] = {
            "optimal_k": int(elbow_point) if elbow_point is not None else None,
            "analysis": analysis
        }

    with open("results/cluster_analysis.json", "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)



if __name__ == "__main__":
    main()