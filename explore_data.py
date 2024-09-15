import json
from icecream import ic


def count_items():
    with open("data/tunes.json") as f:
        tunes = json.load(f)

    with open("data/sets.json") as f:
        sets = json.load(f)
    for tuneset in sets:
        if tuneset["tuneset"] == "76481":
            print(tuneset)


    all_sets = []
    for item in sets:
        all_sets.append(item["tuneset"])


def main():
    count_items()


main()
