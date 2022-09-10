import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from meatnarrative_clf import DataCleanerEnglish
from meatnarrative_clf import DataCleanerGerman

DIR_GER_DATA = "meat_narratives/data/"
DIR_ENG_DATA = "meat_narratives/data/AW__Coded_files/"

statement_type_dict = {"narrative": 0, "goal": 1, "instrument": 2}

statement_topic_dict = {"meat": 0, "substitute": 1, "plant based": 2, "all": 3}

topic_valence_dict = {
    "pro": 0,
    "contra": 1,
}

statement_reference_dict = {
    "health": 0,
    "environment": 1,
    "climate": 2,
    "biodiversity": 3,
    "land usage": 4,
    "water usage and quality": 5,
    "deforestation": 6,
    "animal welfare": 7,
    "working conditions": 8,
    "pandemics and epizootic diseases": 9,
    "antibiotics": 10,
    "economy": 11,
    "moral and ethic": 12,
    "taste and texture": 13,
    "world food supply": 14,
    "highly processed": 15,
    "social fairness": 16,
    "oligopoly": 17,
    "tradition and culture": 18,
    "food security": 19,
    "social inequality": 20,
}


def split_title_to_date_year(str):
    full_date = str.split(" ")[0]
    year = full_date.split("-")[0]
    return year


# Make the actual plots
def get_data(args):
    if args.language == "german":
        frame = DataCleanerGerman.generate_dataset(DIR_GER_DATA)
    elif args.language == "english":
        frame = DataCleanerEnglish.generate_dataset(DIR_ENG_DATA)
    else:
        print("Language not supported")
        return

    # Split the document title column into date
    frame["year"] = frame["document title"].apply(
        lambda row: split_title_to_date_year(row)
    )
    frame.drop(["document title"], axis=1, inplace=True)

    return frame


def make_plots(frame, args):

    new_frame = frame.groupby(["year"])
    type_count = new_frame["statement_type"].value_counts()

    # Plot a stacked bar chart for the statement type
    type_count.unstack().plot(kind="bar", stacked=True, figsize=(10, 10))
    plt.title(
        "Statement type over time for the {} case".format(args.language.capitalize()),
        fontsize=20,
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of statements", fontsize=14)
    # Axes label sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Legend location
    plt.legend(loc="upper left")
    # Manual legend size for English
    if args.language == "english":
        plt.legend(loc="upper left", prop={"size": 12})
    plt.savefig("meat_narratives/plots/{}_statement_type.pdf".format(args.language))

    # Plot a stacked bar chart for the statement topic
    topic_count = new_frame["statement_topic"].value_counts()
    topic_count.unstack().plot(kind="bar", stacked=True, figsize=(10, 10))
    plt.title(
        "Statement topic over time for the {} case".format(args.language.capitalize()),
        fontsize=20,
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of statements", fontsize=14)
    # Axes label sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="upper left")
    # Manual legend size for English
    if args.language == "english":
        plt.legend(loc="upper left", prop={"size": 12})
    plt.savefig("meat_narratives/plots/{}_statement_topic.pdf".format(args.language))

    # Plot a stacked bar chart for the topic valence
    valence_count = new_frame["topic_valence"].value_counts()
    valence_count.unstack().plot(kind="bar", stacked=True, figsize=(10, 10))
    plt.title(
        "Topic valence over time for the {} case".format(args.language.capitalize()),
        fontsize=20,
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of statements", fontsize=14)
    # Axes label sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="upper left")
    # Manual legend size for English
    if args.language == "english":
        plt.legend(loc="upper left", prop={"size": 12})
    plt.savefig("meat_narratives/plots/{}_topic_valence.pdf".format(args.language))

    # Plot a stacked bar chart for the statement reference
    reference_count = new_frame["statement_reference"].value_counts()
    ax = reference_count.unstack().plot(kind="bar", stacked=True, figsize=(150, 100))
    plt.title(
        "Statement reference over time for the {} case".format(
            args.language.capitalize()
        ),
        fontsize=200,
    )
    plt.xlabel("Year", fontsize=120)
    plt.ylabel("Number of statements", fontsize=120)
    # Axes label sizes
    plt.xticks(fontsize=120)
    plt.yticks(fontsize=120)
    plt.legend(fontsize=80)
    plt.legend(loc="upper left", prop={"size": 80})
    # Manual legend size for english
    if args.language == "english":
        plt.legend(loc="upper left", prop={"size": 90})
    plt.savefig(
        "meat_narratives/plots/{}_statement_reference.pdf".format(args.language)
    )

    # Plot a stacked bar chart for the top 5 references
    top_5_reference_count = (
        new_frame["statement_reference"].value_counts().groupby(level=0).nlargest(5)
    )
    ax_5 = top_5_reference_count.unstack().plot(
        kind="bar", stacked=True, figsize=(150, 100)
    )
    plt.title(
        "Top 5 statement references per year for the {} case".format(
            args.language.capitalize()
        ),
        fontsize=200,
    )
    plt.xlabel("Year", fontsize=120)
    plt.ylabel("Number of statements", fontsize=120)
    # Axes label sizes
    plt.xticks(fontsize=120)
    plt.yticks(fontsize=120)
    # Legend font size
    plt.legend(fontsize=90)
    labels = [item.get_text() for item in ax_5.get_xticklabels()]
    # Get all first elements of list of tuples (convert tuples of years to years)
    labels = [item.split(",")[0][1:] for item in labels]
    ax_5.set_xticklabels(labels)
    # ax_5.get_legend().set_bbox_to_anchor((0.96, 0.8))
    plt.legend(loc="upper left", prop={"size": 80})
    # Manual legend size for english
    if args.language == "english":
        plt.legend(loc="upper left", prop={"size": 90})
    plt.savefig(
        "meat_narratives/plots/{}_top_5_statement_reference.pdf".format(args.language)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="german")
    args = parser.parse_args()

    frame = get_data(args)
    make_plots(frame, args)


if __name__ == "__main__":
    main()