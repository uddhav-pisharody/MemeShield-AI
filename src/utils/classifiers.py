import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_classifiers(selected_features: pd.DataFrame, config):
    """
    Train various machine learning classifiers on the selected features.

    Args:
    selected_features (pd.DataFrame): The DataFrame containing selected features and the label.
    config (Config): Configuration object with parameters.

    Returns:
    dict: A dictionary containing trained classifiers.
    """
    # Split the features and labels
    X = selected_features.drop('label', axis=1)
    y = selected_features['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }

    # Train classifiers
    trained_classifiers = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        trained_classifiers[name] = clf
        print(f"{name} classifier trained.")

    # Example: pickle.dump(trained_classifiers, open(config.classifiers_save_path, 'wb'))

    return trained_classifiers

def evaluate_classifiers(classifiers: dict, config):
    """
    Evaluate the trained classifiers on the test data.

    Args:
    classifiers (dict): A dictionary containing trained classifiers.
    config (Config): Configuration object with parameters.
    
    Returns:
    None
    """
    selected_features = pd.read_csv(config.test_features_path)  
    X_test = selected_features.drop('label', axis=1)
    y_test = selected_features['label']

    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.2f}")
        print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
