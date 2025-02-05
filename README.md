# T-shirt Demand Prediction

**Author**: Weronika Łoś (266535)

## Project Objective
The goal of this project is to analyze a dataset of t-shirts with various attributes and predict the demand for t-shirts using machine learning algorithms. The steps include data exploration, data preparation, classification, and evaluation.

## Steps

### 1. Data Exploration
The dataset contains 20,000 entries with attributes like size, material, color, sleeves, and demand.  
- **Target Column (demand)**: "low", "medium", "high"  
- **Label Distribution**: The dataset is imbalanced, with fewer "low" demand entries.

### 2. Data Preparation
Categorical data (ordinal and nominal) were converted into numerical values:  
- **Sleeves**: short → 0, long → 1  
- **Size**: XS → 0, S → 1, ..., 3XL → 6  
One-hot encoding was applied, and the data was split into training and testing sets.

### 3. Classification and Evaluation
The following metrics were calculated for the Naive Bayes and Decision Tree models under three preprocessing conditions: no preprocessing, normalization, and PCA.

#### Naive Bayes:
| Metric      | No Preprocessing | Normalization | PCA     |
|-------------|------------------|---------------|---------|
| Accuracy   | 0.63             | 0.59          | 0.63    |
| Recall     | 0.56             | 0.54          | 0.60    |
| F1-score   | 0.57             | 0.51          | 0.58    |
| Precision  | 0.76             | 0.66          | 0.58    |

#### Decision Tree:
| Metric      | No Preprocessing | Normalization | PCA     |
|-------------|------------------|---------------|---------|
| Accuracy   | 0.97             | 0.97          | 0.58    |
| Recall     | 0.96             | 0.96          | 0.62    |
| F1-score   | 0.96             | 0.96          | 0.58    |
| Precision  | 0.96             | 0.96          | 0.59    |

### Findings:
- **Naive Bayes** and **Decision Tree**: Best performance was achieved after preprocessing. 
- **Decision Tree** outperformed **Naive Bayes** in all metrics.
- **Learning Curve**: The Decision Tree showed overfitting, while Naive Bayes demonstrated better generalization.

### Hyperparameter Tuning for Decision Tree:
- **Max Depth**: Increasing depth improves performance until it reaches a plateau around depth 10.
- **Min Samples Leaf**: Smaller values of `min_samples_leaf` allow the tree to better fit data, improving performance.
- **Max Leaf Nodes**: More leaf nodes help the model split data more effectively, with results stabilizing around 24-30 nodes.

### Conclusion:
The decision tree, with appropriate preprocessing and hyperparameter tuning, proved to be the most effective model for predicting t-shirt demand.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (for visualizations, if any)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/t-shirt-demand-prediction.git
   cd t-shirt-demand-prediction
