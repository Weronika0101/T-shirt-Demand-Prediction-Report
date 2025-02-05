import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['sleeves'] = df['sleeves'].replace({'short': 0, 'long': 1})
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL']
    df['size'] = df['size'].replace({size: i for i, size in enumerate(sizes)})
    dummies = pd.get_dummies(df[['material', 'color']])
    df = pd.concat([df, dummies], axis=1)
    df.drop(['material', 'color'], axis=1, inplace=True)
    return df


def plot_hyperparameter_tuning(results, param_name):
    plt.figure(figsize=(12, 8))
    plt.plot(results[param_name], results['accuracy'], label='Accuracy', marker='o')
    plt.plot(results[param_name], results['precision'], label='Precision', marker='o')
    plt.plot(results[param_name], results['recall'], label='Recall', marker='o')
    plt.plot(results[param_name], results['f1'], label='F1 Score', marker='o')

    plt.title(f"'{param_name}' parameter tuning")
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_hyperparameter(filepath, param_name, param_values):
    df = load_and_preprocess_data(filepath)
    X = df.drop(['demand'], axis=1)
    y = df['demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    results = []
    for value in param_values:
        params = {param_name: value}
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append({param_name: value, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})
    results_df = pd.DataFrame(results)
    plot_hyperparameter_tuning(results_df, param_name)
    return results_df


if __name__ == '__main__':
    max_depth_values = range(1, 20)
    results_max_depth = test_hyperparameter('t-shirts.csv', 'max_depth', max_depth_values)

    min_samples_leaf_values = range(1, 20)
    results_min_samples_leaf = test_hyperparameter('t-shirts.csv', 'min_samples_leaf', min_samples_leaf_values)

    max_leaf_nodes_values = range(2, 40)
    results_max_leaf_nodes = test_hyperparameter('t-shirts.csv', 'max_leaf_nodes', max_leaf_nodes_values)

    print("Results for max_depth:")
    print(results_max_depth)
    print("\nResults for min_samples_leaf:")
    print(results_min_samples_leaf)
    print("\nResults for max_leaf_nodes:")
    print(results_max_leaf_nodes)
