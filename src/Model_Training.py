import pickle as pk
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

def pickle_dump(file, name):
    # Save a Python object (model/vectorizer) to a pickle file
    output_dir = r"D:\Guvi_Project\Personalized Learning Assistant\models\ml_models"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
    vector_filename = os.path.join(output_dir, f"{name}.pkl")
    with open(vector_filename, 'wb') as f:
        pk.dump(file, f)  # Serialize and save object
    print(f"{name} Model saved successfully.")

def pickle_load(file):
    # Load a Python object from a pickle file
    input_dir = r"D:\Guvi_Project\Personalized Learning Assistant\models\ml_models"
    vector_filename = os.path.join(input_dir, f"{file}.pkl")
    try:
        with open(vector_filename, 'rb') as f:
            loaded_model = pk.load(f)  # Deserialize and load object
        return loaded_model
    except FileNotFoundError:
        print(f"Error: The model file '{vector_filename}' does not exist.")
        return None
    except Exception as e:
        print(f"Error loading model '{file}': {e}")
        return None

def vectorization(data, name):
    # Fit a TF-IDF vectorizer on text data, save it, and return transformed data as DataFrame
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(data)
    pickle_dump(tfidf, name)  # Save fitted vectorizer
    # Convert sparse matrix to dense DataFrame with feature names as columns
    return pd.DataFrame(vectors.toarray(), columns=tfidf.get_feature_names_out())

def classification(model, X, y, name, test_size=0.2, random_state=42):
    # Train classification model, evaluate, and save trained model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pickle_dump(model, name)  # Save model
    # Calculate evaluation metrics and return as DataFrame
    eval_scores = pd.DataFrame([{
        "Model": model.__class__.__name__,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall Score": recall_score(y_test, y_pred),
        "Precision Score": precision_score(y_test, y_pred),
        "f1 Score": f1_score(y_test, y_pred)
    }])
    return eval_scores

def regression(model, X, y, name, test_size=0.2, random_state=42):
    # Train regression model, evaluate, and save trained model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pickle_dump(model, name)  # Save model
    # Calculate regression evaluation metrics and return as DataFrame
    eval_scores = pd.DataFrame([{
        "Model": model.__class__.__name__,
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred)
    }])
    return eval_scores
