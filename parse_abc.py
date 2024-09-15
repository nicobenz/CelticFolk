import re
from collections import Counter
from music21 import converter, interval, pitch, stream, environment, abcFormat
import json
from typing import Dict, List
from collections import Counter
import pandas as pd
import warnings


def parse_abc_notation(abc_string):
    # Remove header information (lines starting with a letter followed by a colon)
    melody_lines = [line for line in abc_string.split('\r\n') if not re.match(r'^[A-Z]:', line)]

    # Join the remaining lines
    melody_string = ' '.join(melody_lines)

    # Remove bar lines and whitespace
    melody_string = re.sub(r'[|\s]', '', melody_string)

    return melody_string


def extract_intervals(melody_string):
    score = converter.parse(f"tinynotation: {melody_string}")
    pitches = [p for p in score.pitches]
    intervals = []
    for i in range(len(pitches) - 1):
        intervals.append(interval.Interval(pitches[i], pitches[i + 1]).name)
    return intervals


def analyze_melody(abc_string):
    try:
        # Add a default header if it's missing
        if not abc_string.startswith('X:'):
            abc_string = f"X:1\nM:4/4\nL:1/8\nK:C\n{abc_string}"

        # Parse the ABC notation using converter
        score = converter.parse(abc_string, format='abc')

        # Extract pitches
        pitches = [p for p in score.pitches]

        if not pitches:
            raise ValueError("No pitches found in the melody")

        # Extract intervals
        intervals = []
        for i in range(len(pitches) - 1):
            intervals.append(interval.Interval(pitches[i], pitches[i + 1]).name)

        # Analyze interval distribution
        interval_dist = Counter(intervals)

        # Analyze pitch range
        pitch_range = interval.Interval(pitches[0], pitches[-1])

        # Analyze most common starting and ending notes
        start_note = pitches[0].name
        end_note = pitches[-1].name

        return {
            'interval_distribution': dict(interval_dist),
            'pitch_range': abs(pitch_range.semitones),
            'start_note': start_note,
            'end_note': end_note
        }
    except Exception as e:
        raise ValueError(f"Error analyzing melody: {str(e)}")


def compare_melodies(abc1, abc2):
    try:
        analysis1 = analyze_melody(abc1)
        analysis2 = analyze_melody(abc2)
    except Exception as e:
        raise ValueError(f"Error comparing melodies: {str(e)}")

    # Compare interval distributions
    common_intervals = set(analysis1['interval_distribution'].keys()) & set(analysis2['interval_distribution'].keys())
    interval_similarity = sum(
        min(analysis1['interval_distribution'][i], analysis2['interval_distribution'][i]) for i in common_intervals) / \
                          max(sum(analysis1['interval_distribution'].values()),
                              sum(analysis2['interval_distribution'].values()))

    # Compare pitch ranges
    range_overlap = min(analysis1['pitch_range'], analysis2['pitch_range']) / max(analysis1['pitch_range'], analysis2['pitch_range'])

    # Compare start and end notes
    start_end_match = (analysis1['start_note'] == analysis2['start_note']) + (
                analysis1['end_note'] == analysis2['end_name'])

    return {
        'interval_similarity': interval_similarity,
        'range_overlap': range_overlap,
        'start_end_match': start_end_match / 2  # Normalize to [0, 1]
    }


def generate_melodic_features(tunes: List[Dict], sets: List[List[int]]):
    """
    Generate melodic features for tunes and sets.

    :param tunes: List of dictionaries, each containing 'id' and 'abc' keys
    :param sets: List of lists, each inner list contains tune IDs for a set
    :return: Two DataFrames - one for tune features, one for transition features
    """
    # Assuming analyze_melody and compare_melodies functions are available

    # Generate features for individual tunes
    tune_features = []
    for tune in tunes:
        analysis = analyze_melody(tune['abc'])
        features = {
            'tune_id': tune['id'],
            'pitch_range': int(analysis['pitch_range'].split(' ')[0]),  # Convert "X semitones" to int
            'start_note': analysis['start_note'],
            'end_note': analysis['end_note'],
            'most_common_interval': max(analysis['interval_distribution'], key=analysis['interval_distribution'].get),
            'interval_variety': len(analysis['interval_distribution'])
        }
        tune_features.append(features)

    tune_df = pd.DataFrame(tune_features)

    # Generate features for transitions between tunes
    transition_features = []
    for set_id, set_tunes in enumerate(sets):
        for i in range(len(set_tunes) - 1):
            tune1 = next(tune for tune in tunes if tune['id'] == set_tunes[i])
            tune2 = next(tune for tune in tunes if tune['id'] == set_tunes[i + 1])
            comparison = compare_melodies(tune1['abc'], tune2['abc'])
            features = {
                'set_id': set_id,
                'position': i,
                'tune1_id': set_tunes[i],
                'tune2_id': set_tunes[i + 1],
                'interval_similarity': comparison['interval_similarity'],
                'range_overlap': comparison['range_overlap'],
                'start_end_match': comparison['start_end_match']
            }
            transition_features.append(features)

    transition_df = pd.DataFrame(transition_features)

    return tune_df, transition_df