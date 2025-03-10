import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess the data
def preprocess_data(df):
    # Convert date columns
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format='%d/%m/%Y')
    
    # Feature engineering
    # Add more preprocessing steps as needed
    
    return df

# Exploratory data analysis
def perform_eda(df):
    # Transaction counts by category
    category_counts = df['Classification_Tag'].value_counts()
    
    # Amount distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=30)
    plt.title('Transaction Amount Distribution')
    plt.savefig('amount_distribution.png')
    
    # More EDA visualizations can be added
    
    return category_counts

# Model training and evaluation
def train_models(df):
    # Prepare features and target
    X = df['Description']
    y = df['Classification_Tag']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted')
        }
    
    return results

# Main function
def main():
    # Load data
    df = load_data('sample_transactions_DS (1).csv')
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Perform EDA
    category_counts = perform_eda(df)
    print("Transaction counts by category:")
    print(category_counts)
    
    # Train and evaluate models
    model_results = train_models(df)
    
    # Display model results
    for model_name, metrics in model_results.items():
        print(f"\n{model_name} Performance:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
