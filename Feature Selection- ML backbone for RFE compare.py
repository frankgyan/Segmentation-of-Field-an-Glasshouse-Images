"""
Comparative Analysis of Recursive Feature Elimination (RFE) with Various Base Estimators

This script evaluates the performance of RFE feature selection when paired with different
base estimators, all used in conjunction with a DecisionTreeClassifier for final modeling.
"""

import pandas as pd
from numpy import mean, std
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class RFEEvaluator:
    """
    A class to evaluate RFE performance with different base estimators.

    Attributes:
        models (dict): Dictionary of model pipelines to evaluate
        random_state (int): Random seed for reproducibility
        n_features_to_select (int): Number of features to select in RFE
    """

    def __init__(self, random_state=42, n_features_to_select=10):
        """Initialize the evaluator with random state and feature selection parameters."""
        self.random_state = random_state
        self.n_features_to_select = n_features_to_select
        self.models = self._initialize_models()

    def _initialize_models(self):
        """
        Create a dictionary of model pipelines pairing RFE with different estimators.

        Returns:
            dict: Dictionary of {model_name: pipeline} pairs
        """
        models = {}

        # Base estimators for RFE
        estimators = {
            'LogisticRFE': LogisticRegression(random_state=self.random_state),
            'PerceptronRFE': Perceptron(random_state=self.random_state),
            'TreeRFE': DecisionTreeClassifier(random_state=self.random_state),
            'ForestRFE': RandomForestClassifier(random_state=self.random_state),
            'BoostedRFE': GradientBoostingClassifier(random_state=self.random_state)
        }

        for name, estimator in estimators.items():
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=self.n_features_to_select
            )
            models[name] = Pipeline([
                ('feature_selection', rfe),
                ('classifier', DecisionTreeClassifier(random_state=self.random_state))
            ])

        return models

    def evaluate_models(self, X, y):
        """
        Evaluate all models using repeated stratified k-fold cross-validation.

        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector

        Returns:
            tuple: (results, names) where results contains scores and names contains model names
        """
        results, names = [], []
        cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=3,
            random_state=self.random_state
        )

        for name, model in self.models.items():
            try:
                scores = cross_val_score(
                    model, X, y,
                    scoring='accuracy',
                    cv=cv,
                    n_jobs=-1,
                    error_score='raise'
                )
                results.append(scores)
                names.append(name)
                print(f'{name:>12} | Mean Accuracy: {mean(scores):.3f} (±{std(scores):.3f})')
            except Exception as e:
                print(f'Error evaluating {name}: {str(e)}')
                continue

        return results, names

    def plot_results(self, results, names):
        """Create a boxplot visualization of model performance."""
        if not results or not names:
            print("No results to plot")
            return

        pyplot.figure(figsize=(10, 6))
        pyplot.boxplot(results, labels=names, showmeans=True, patch_artist=True)
        pyplot.title('Model Comparison: RFE with Different Base Estimators', pad=20)
        pyplot.ylabel('Accuracy')
        pyplot.xticks(rotation=45)
        pyplot.tight_layout()
        pyplot.show()


def load_data_from_excel(excel_file_path):
    """
    Load data from Excel file and separate features and target.

    Args:
        excel_file_path (str): Path to the Excel file

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_excel(excel_file_path)
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]   # Last column as target

    # Basic data validation
    if X.empty or y.empty:
        raise ValueError("Empty dataset after loading from Excel")

    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y

# --- Code to execute in the notebook ---

# Load data from the previously defined excel_file_path
# X and y are already available from previous cells
# X, y = load_data_from_excel(excel_file_path) # Uncomment and use this if X and y are not in the global scope

# Set n_features_to_select as 30% of total features (rounded up)
n_features = max(1, int(0.3 * X.shape[1]))

# Initialize and run evaluation
evaluator = RFEEvaluator(random_state=42, n_features_to_select=n_features)
results, names = evaluator.evaluate_models(X, y)

if results and names:
    evaluator.plot_results(results, names)
else:
    print("No valid results to display")
