import pandas as pd
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
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


def plot_learning_curve(estimator, X, y, title, ax, preprocessing_tool=None):
    if preprocessing_tool is not None:
        X = preprocessing_tool.fit_transform(X)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy Score")
    ax.grid()

    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score")

    ax.legend(loc="best")


def rate_classification(X_train, X_test, y_train, y_test, model, preprocessing_tool=None):
    print(f'MODEL: {model}; PREPROCESSING: {preprocessing_tool}')
    if preprocessing_tool is not None:
        X_train = preprocessing_tool.fit_transform(X_train)
        X_test = preprocessing_tool.fit_transform(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap=plt.cm.Greens)
    plt.show()
    return accuracy, precision, recall, f1


def visualize_preprocessing(X, preprocessing_tool, title):
    X_transformed = preprocessing_tool.fit_transform(X)
    df_transformed = pd.DataFrame(X_transformed)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    seaborn.heatmap(df_transformed, annot=False, cmap='viridis')
    plt.show()
    return df_transformed.head(10)


def main(filepath):
    df = load_and_preprocess_data(filepath)
    X = df.drop(['demand'], axis=1)
    y = df['demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    models = [
        (GaussianNB(), None),
        (GaussianNB(), Normalizer()),
        (GaussianNB(), PCA()),
        (DecisionTreeClassifier(), None),
        (DecisionTreeClassifier(), Normalizer()),
        (DecisionTreeClassifier(), PCA())
    ]

    results = []

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    for i, (model, preprocessing_tool) in enumerate(models):
        accuracy, precision, recall, f1 = rate_classification(X_train, X_test, y_train, y_test, model,
                                                              preprocessing_tool)
        results.append({
            'model': model.__class__.__name__,
            'preprocessing': preprocessing_tool.__class__.__name__ if preprocessing_tool else 'None',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        title = f'Learning Curve for {model.__class__.__name__} with {preprocessing_tool}'
        plot_learning_curve(model, X, y, title, axes[i], preprocessing_tool)

    fig.tight_layout()
    fig.show()

    results_df = pd.DataFrame(results)
    mean_results = results_df.groupby(['model', 'preprocessing']).mean().reset_index()
    print(mean_results)

    return results_df, mean_results


if __name__ == "__main__":
    results_df, mean_results = main('t-shirts.csv')

    print("Mean Results:")
    print(mean_results)

    normalizer = Normalizer()
    pca = PCA()

    norm_df = visualize_preprocessing(results_df.drop(['model', 'preprocessing'], axis=1), normalizer, "Normalizacja")
    pca_df = visualize_preprocessing(results_df.drop(['model', 'preprocessing'], axis=1), pca, "PCA")

    print("Normalized Data:")
    print(norm_df)

    print("\nPCA Data:")
    print(pca_df)

