import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from meatnarrative_clf import DataCleanerEnglish, DataCleanerGerman

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
    color_list = sns.color_palette("cubehelix", 5)
    add_color_list = sns.color_palette("dark", 5)
    color_list.extend(add_color_list)
    add_color_list = sns.color_palette("Paired")[::-1][:5]
    color_list.extend(add_color_list)
    add_color_list = sns.color_palette("colorblind", 2)
    color_list.extend(add_color_list)

    ax = reference_count.unstack().plot(
        kind="bar",
        stacked=True,
        figsize=(150, 100),
        # Give me a list of 17 unique colors
        color=color_list,
    )
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
        kind="bar",
        stacked=True,
        figsize=(150, 100),
        # Tab color list for matplotlib
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "gold",
        ],
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


def make_more_plots(frame, args):
    # Group frame statement type and split into sub frames
    grouped_frame = frame.groupby("statement_topic")
    # Split grouped frame
    for name, group in grouped_frame:
        # Groupby year
        group = group.groupby("year")
        # Plot a stacked bar chart for topic valence
        valence_count = group["topic_valence"].value_counts()
        valence_count.unstack().plot(kind="bar", stacked=True, figsize=(10, 10))
        plt.title(
            "Topic valence shifts for the statement topic: {}".format(name.upper()),
            fontsize=20,
        )
        plt.xlabel("Year", fontsize=14)
        plt.ylabel("Number of statements", fontsize=14)
        # Axes label sizes
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="upper left", fontsize=14)
        # Manual legend size for English
        if args.language == "english":
            plt.legend(loc="upper left", prop={"size": 16})
        plt.savefig(
            "meat_narratives/plots/{}_lang_topic_valence_shifts_for_topic_{}.pdf".format(
                args.language, name
            )
        )

        # Plot a stacked bar chart for top 5 statement references
        top_5_reference_count = (
            group["statement_reference"].value_counts().groupby(level=0).nlargest(5)
        )
        _ax_5 = top_5_reference_count.unstack().plot(
            kind="bar",
            stacked=True,
            figsize=(150, 100),
            # Tab color list for matplotlib
            color=[
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
                "gold",
                "black",
                "#ccffcc",
            ],
        )
        plt.title(
            "Top 5 statement references shifts for the statement topic: {}".format(
                name.upper()
            ),
            fontsize=200,
        )
        plt.xlabel("Year", fontsize=140)
        plt.ylabel("Number of statements", fontsize=140)
        # Axes label sizes
        plt.xticks(fontsize=140)
        plt.yticks(fontsize=140)
        # Legend font size
        plt.legend(fontsize=120)
        labels = [item.get_text() for item in _ax_5.get_xticklabels()]
        # Get all first elements of list of tuples (convert tuples of years to years)
        labels = [item.split(",")[0][1:] for item in labels]
        _ax_5.set_xticklabels(labels)
        # ax_5.get_legend().set_bbox_to_anchor((0.96, 0.8))
        plt.legend(loc="upper left", prop={"size": 80})
        # Manual legend size for english
        if args.language == "english":
            plt.legend(loc="upper left", prop={"size": 80})
        plt.savefig(
            "meat_narratives/plots/{}_lang_top_5_statement_reference_for_topic_{}.pdf".format(
                args.language, name
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="german")
    args = parser.parse_args()

    frame = get_data(args)
    make_plots(frame, args)
    make_more_plots(frame, args)


if __name__ == "__main__":
    main()
