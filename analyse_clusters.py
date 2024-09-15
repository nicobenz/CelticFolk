import json
from scipy.stats import chi2_contingency
import numpy as np


def calculate_baseline(cluster_sizes):
    total_songs = sum(cluster_sizes.values())
    probability = sum((size / total_songs) ** 2 for size in cluster_sizes.values())
    return probability


def perform_chi_square_test(observed, expected):
    chi2, p_value, dof, _ = chi2_contingency([observed, expected])
    return chi2, p_value, dof


def analyze_sets_of_two(data):
    cluster_sizes = data['cluster_distribution']
    total_sets = data['total_sets']
    observed_same_cluster = data['same_cluster']

    baseline_prob = calculate_baseline(cluster_sizes)
    expected_same_cluster = total_sets * baseline_prob

    observed = np.array([observed_same_cluster, total_sets - observed_same_cluster])
    expected = np.array([expected_same_cluster, total_sets - expected_same_cluster])

    chi2, p_value, dof = perform_chi_square_test(observed, expected)

    return {
        'baseline_probability': baseline_prob,
        'chi_square_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof
    }


def analyze_sets_of_three(data):
    cluster_sizes = data['cluster_distribution']
    total_sets = data['total_sets']
    all_same_cluster = data['all_same_cluster']
    two_same_cluster = data['two_same_cluster']

    baseline_prob = calculate_baseline(cluster_sizes)

    prob_all_same = baseline_prob
    prob_exactly_two_same = 3 * baseline_prob * (1 - baseline_prob)
    prob_at_least_two_same = prob_all_same + prob_exactly_two_same

    # Test for all three in same cluster
    expected_all_same = total_sets * prob_all_same
    observed_all_same = np.array([all_same_cluster, total_sets - all_same_cluster])
    expected_all_same_array = np.array([expected_all_same, total_sets - expected_all_same])
    chi2_all, p_value_all, dof_all = perform_chi_square_test(observed_all_same, expected_all_same_array)

    # Test for exactly two in same cluster
    expected_exactly_two_same = total_sets * prob_exactly_two_same
    observed_exactly_two_same = np.array([two_same_cluster, total_sets - two_same_cluster])
    expected_exactly_two_same_array = np.array([expected_exactly_two_same, total_sets - expected_exactly_two_same])
    chi2_exactly_two, p_value_exactly_two, dof_exactly_two = perform_chi_square_test(observed_exactly_two_same,
                                                                                     expected_exactly_two_same_array)

    # Test for at least two in same cluster
    expected_at_least_two_same = total_sets * prob_at_least_two_same
    observed_at_least_two_same = np.array(
        [all_same_cluster + two_same_cluster, total_sets - all_same_cluster - two_same_cluster])
    expected_at_least_two_same_array = np.array([expected_at_least_two_same, total_sets - expected_at_least_two_same])
    chi2_two, p_value_two, dof_two = perform_chi_square_test(observed_at_least_two_same,
                                                             expected_at_least_two_same_array)

    return {
        'baseline_probability': baseline_prob,
        'full_match': {
            'baseline_probability': prob_all_same,
            'chi_square_statistic': chi2_all,
            'p_value': p_value_all,
            'degrees_of_freedom': dof_all
        },
        'patial_match': {
            'baseline_probability': prob_exactly_two_same,
            'chi_square_statistic': chi2_exactly_two,
            'p_value': p_value_exactly_two,
            'degrees_of_freedom': dof_exactly_two
        },
        'combined_matches': {
            'baseline_probability': prob_at_least_two_same,
            'chi_square_statistic': chi2_two,
            'p_value': p_value_two,
            'degrees_of_freedom': dof_two
        }
    }


# Load the data
with open('results/cluster_analysis.json', 'r') as f:
    data = json.load(f)

# Analyze sets of two
results_two = analyze_sets_of_two(data['sets_of_two']['analysis'])

# Analyze sets of three
results_three = analyze_sets_of_three(data['sets_of_three']['analysis'])

# Prepare the results dictionary
distribution_analysis = {
    'sets_of_two': results_two,
    'sets_of_three': results_three
}

# Save the results to a JSON file
with open('results/distribution_analysis.json', 'w') as f:
    json.dump(distribution_analysis, f, indent=2)

print("Analysis complete. Results saved to results/distribution_analysis.json")