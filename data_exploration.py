import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    return pd.read_csv(filepath)


def add_value_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 10), textcoords='offset points')


def plot_distribution(data, column, title, ax):
    sns.countplot(x=column, data=data, ax=ax)
    add_value_labels(ax)
    ax.set_title(title)


def display_statistics(data):
    data_info = data.describe(include='all')
    print("Basic Statistics:")
    print(data_info)


def main(filepath):

    data = load_data(filepath)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    plot_distribution(data, 'size', 'Size Distribution', axs[0, 0])
    plot_distribution(data, 'material', 'Material Distribution', axs[0, 1])
    plot_distribution(data, 'color', 'Color Distribution', axs[1, 0])
    plot_distribution(data, 'sleeves', 'Sleeves Distribution', axs[1, 1])

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='demand', data=data)
    add_value_labels(ax)
    plt.title('Demand Distribution')
    plt.show()

    display_statistics(data)


if __name__ == "__main__":
    main('t-shirts.csv')

