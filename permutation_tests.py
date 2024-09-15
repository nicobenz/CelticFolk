import json
from itertools import chain, combinations
import numpy as np
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp


def reduce_data():
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

    for k, v in set_collection.items():
        v.sort(key=lambda x: x["position"])
        for tune in v:
            tune.pop("position")

    tune_set_2 = []
    tune_set_3 = []
    for k, v in set_collection.items():
        if len(v) == 2:
            tune_set_2.append(v)
        if len(v) == 3:
            tune_set_3.append(v)

    return [tune_set_2, tune_set_3]


def get_relative_major(tonic, mode):
    mode_to_major = {
        'major': 0,
        'dorian': -2,
        'phrygian': -4,
        'lydian': 5,
        'mixolydian': -1,
        'minor': -3,
        'locrian': -5
    }
    circle = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab']
    tonic_index = circle.index(tonic)
    relative_major_index = (tonic_index + mode_to_major.get(mode.lower(), 0)) % 12
    return circle[relative_major_index]

def circle_of_fifths_distance(note1, note2):
    circle = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab']
    index1 = circle.index(note1)
    index2 = circle.index(note2)
    return min((index1 - index2) % 12, (index2 - index1) % 12)

def test_statistic_circle_of_fifths(sets):
    total_score = 0
    for tune_set in sets:
        tonics = [tune['tonic'] for tune in tune_set]
        distances = [circle_of_fifths_distance(t1, t2) for t1, t2 in combinations(tonics, 2)]
        set_score = np.mean(distances) if distances else 0
        total_score += set_score

    return total_score / len(sets) if sets else 0

def mode_similarity(mode1, mode2):
    mode_intervals = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'locrian': [0, 1, 3, 5, 6, 8, 10]
    }
    set1 = set(mode_intervals.get(mode1.lower(), []))
    set2 = set(mode_intervals.get(mode2.lower(), []))
    return len(set1.intersection(set2)) / len(set1.union(set2))

def test_statistic_mode_similarity(sets):
    total_score = 0
    for tune_set in sets:
        modes = [tune['mode'] for tune in tune_set]
        similarities = [mode_similarity(m1, m2) for m1, m2 in combinations(modes, 2)]
        set_score = np.mean(similarities) if similarities else 0
        total_score += set_score

    return total_score / len(sets) if sets else 0


def test_statistic_entropy(sets, features):
    def attribute_diversity(attribute_list):
        counts = Counter(attribute_list)
        probabilities = [count / len(attribute_list) for count in counts.values()]
        entropy = -sum(p * np.log(p) for p in probabilities)  # Shannon entropy
        return entropy

    def tonic_spread(tonic_values):
        circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab']
        indices = [circle_of_fifths.index(tonic) for tonic in tonic_values]
        spread = np.std(indices)
        return spread

    total_score = 0
    for tune_set in sets:
        set_score = 0
        for feature in features:
            if feature == 'tonic':
                values = [tune[feature] for tune in tune_set]
                similarity = 1 / (1 + tonic_spread(values))
            else:
                values = [tune[feature] for tune in tune_set]
                similarity = 1 / (1 + attribute_diversity(values))
            set_score += similarity

        set_score /= len(features)
        total_score += set_score

    return total_score / len(sets) if sets else 0


def test_statistic_jaccard(sets, features):
    def jaccard_similarity(set1, set2):
        intersection = len(set(set1).intersection(set(set2)))
        union = len(set(set1).union(set(set2)))
        return intersection / union if union > 0 else 0

    total_score = 0
    for tune_set in sets:
        set_score = 0
        comparisons = 0
        for tune1, tune2 in combinations(tune_set, 2):
            feature_similarity = sum(jaccard_similarity([tune1[f]], [tune2[f]]) for f in features)
            set_score += feature_similarity / len(features)
            comparisons += 1

        total_score += set_score / comparisons if comparisons > 0 else 0

    return total_score / len(sets) if sets else 0


def test_statistic_chi_square(sets, features):
    def calculate_overall_frequencies(all_attr):
        overall_counts = Counter(all_attr)
        total = sum(overall_counts.values())
        return {attr: count / total for attr, count in overall_counts.items()}

    all_attributes = {f: [tune[f] for tune_set in sets for tune in tune_set] for f in features}
    overall_freq = {f: calculate_overall_frequencies(attrs) for f, attrs in all_attributes.items()}

    def chi_square_test(attribute_list, overall_frequency):
        observed = Counter(attribute_list)
        n = len(attribute_list)

        all_categories = set(overall_frequency.keys()) | set(observed.keys())
        observed_array = np.array([observed.get(cat, 0) for cat in all_categories])
        expected_array = np.array([overall_frequency.get(cat, 0) * n for cat in all_categories])

        expected_array = np.maximum(expected_array, 0.01)

        chi2 = np.sum((observed_array - expected_array) ** 2 / expected_array)
        return chi2

    total_score = 0
    for tune_set in sets:
        set_score = sum(chi_square_test([tune[f] for tune in tune_set], overall_freq[f]) for f in features)
        total_score += set_score / len(features)

    return total_score / len(sets) if sets else 0


def permutation_testing(tqdm_label, tune_set, test_statistic, features=None, n_resamples=10_000):
    all_tunes = list(chain.from_iterable(tune_set))

    # Check if the test_statistic function expects features
    if 'features' in test_statistic.__code__.co_varnames:
        actual_statistic = test_statistic(tune_set, features)
    else:
        actual_statistic = test_statistic(tune_set)

    permuted_statistics = []

    for _ in tqdm(range(n_resamples), desc=tqdm_label):
        np.random.shuffle(all_tunes)
        start = 0
        permuted_sets = []
        for set_size in [len(s) for s in tune_set]:
            permuted_sets.append(all_tunes[start:start + set_size])
            start += set_size

        # Check if the test_statistic function expects features
        if 'features' in test_statistic.__code__.co_varnames:
            permuted_statistic = test_statistic(permuted_sets, features)
        else:
            permuted_statistic = test_statistic(permuted_sets)

        permuted_statistics.append(permuted_statistic)

    p_value = calculate_p_value(actual_statistic, permuted_statistics)

    results = {
        "n_resamples": n_resamples,
        "p_value": p_value,
        "actual_statistic": actual_statistic,
        "min_permuted_statistic": min(permuted_statistics),
        "max_permuted_statistic": max(permuted_statistics),
        "mean_permuted_statistic": np.mean(permuted_statistics),
        "std_dev_permuted_statistics": np.std(permuted_statistics)
    }

    return results


def calculate_p_value(actual_statistic, permuted_statistics):
    n_resamples = len(permuted_statistics)
    if np.mean(permuted_statistics) > actual_statistic:
        return np.sum([stat <= actual_statistic for stat in permuted_statistics]) / n_resamples
    else:
        return np.sum([stat >= actual_statistic for stat in permuted_statistics]) / n_resamples


def process_dataset(dataset_index, clean_data, resamples):
    all_features = ['type', 'meter', 'mode', 'tonic']
    feature_combinations = [all_features] + [[f] for f in all_features]

    save_data = {}
    for features in feature_combinations:
        key = 'all' if len(features) == 4 else features[0]
        save_data[key] = {
            "entropy": permutation_testing(
                f"Entropy (Dataset {dataset_index + 1}) - {key}",
                clean_data,
                test_statistic_entropy,
                features,
                n_resamples=resamples
            ),
            "jaccard_similarity": permutation_testing(
                f"Jaccard (Dataset {dataset_index + 1}) - {key}",
                clean_data,
                test_statistic_jaccard,
                features,
                n_resamples=resamples
            ),
            "chi2_statistics": permutation_testing(
                f"Chi-square (Dataset {dataset_index + 1}) - {key}",
                clean_data,
                test_statistic_chi_square,
                features,
                n_resamples=resamples
            )
        }

        # Add circle of fifths analysis only for tonic
        if key == 'tonic':
            save_data[key]["circle_of_fifths"] = permutation_testing(
                f"Circle of Fifths (Dataset {dataset_index + 1}) - {key}",
                clean_data,
                test_statistic_circle_of_fifths,
                n_resamples=resamples
            )

        # Add mode similarity analysis only for mode
        if key == 'mode':
            save_data[key]["mode_similarity"] = permutation_testing(
                f"Mode Similarity (Dataset {dataset_index + 1}) - {key}",
                clean_data,
                test_statistic_mode_similarity,
                n_resamples=resamples
            )

    return dataset_index, save_data


def main():
    resamples = 10_000
    datasets = reduce_data()

    # threading
    num_cores = 4
    pool = mp.Pool(processes=num_cores)

    # Prepare the arguments for each process
    args = [(i, dataset, resamples) for i, dataset in enumerate(datasets)]

    # Run the processes in parallel
    results = pool.starmap(process_dataset, args)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Organize the results
    save_collection = {
        "two tunes": {},
        "three tunes": {}
    }
    for index, result in results:
        save_collection[list(save_collection.keys())[index]] = result

    # Save the results to a file
    with open("results/permutation_tests.json", "w") as f:
        json.dump(save_collection, f, indent=2)


if __name__ == "__main__":
    main()
