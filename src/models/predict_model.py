# Import accuracy score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# # Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test):
    # Set up a KFold cross-validation
    kfold = KFold(n_splits=5)
    # Use cross-validation to evaluate the model
    scores = cross_val_score(model, X_test_scaled, y_test, cv=kfold)
    # Print the accuracy scores for each fold
    print("Accuracy scores:", scores)
    # Print the mean accuracy and standard deviation of the model
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())

    return scores