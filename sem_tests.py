from semopy import Model, Optimizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from scipy.stats import chi2_contingency
from collections import Counter


def use_sem(data):
    # load to df
    df = pd.DataFrame(data)

    # prepare labeling
    le_type = LabelEncoder()
    le_meter = LabelEncoder()
    le_mode = LabelEncoder()
    le_tonic = LabelEncoder()

    # label process
    df['type'] = le_type.fit_transform(df['type'])
    df['meter'] = le_meter.fit_transform(df['meter'])
    df['mode'] = le_mode.fit_transform(df['mode'])
    df['tonic'] = le_tonic.fit_transform(df['tonic'])

    # describe model parameters
    model_desc = """
    # Measurement model
    Set_Formation =~ meter + type + mode + tonic
    """

    # create model class and load data
    model = Model(model_desc)
    model.load_dataset(df)

    # optimise
    opt = Optimizer(model)
    opt.optimize()

    inspect = model.inspect()
    fit = model.fit()

    result = {
        "inspect": str(inspect),
        "fit": str(fit)
    }


    # Save results to JSON file
    with open("results/sem.json", 'w') as f:
        json.dump(result, f, indent=4)


def chi_square(data, variable="type"):
    df = pd.DataFrame(data)
    contingency_table = pd.crosstab(df['set_id'], df[variable])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-Value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print("Expected Frequencies:")
    print(expected)


def reduce_data():
    with open("data/sets.json") as f:
        data = json.load(f)

    tune_list = []
    for item in data:
        tune = {
            "set_id": item["tuneset"],
            "position": item["settingorder"],
            "name": item["name"],
            "tune_id": item["tune_id"],
            "setting_id": item["setting_id"],
            "type": item["type"],
            "meter": item["meter"],
            "mode": item["mode"][1:],
            "tonic": item["mode"][:1],
            "abc": item["abc"]
        }
        tune_list.append(tune)
    return tune_list


def explore_data(data):
    labels = ["type", "meter", "mode", "tonic"]
    for label in labels:
        types = [item[label] for item in data]

        type_counts = Counter(types)
        print(label, type_counts)


def main():
    reduced_data = reduce_data()
    #explore_data(reduced_data)
    use_sem(reduced_data)
    #chi_square(reduced_data, "type")


main()

