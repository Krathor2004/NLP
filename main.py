import sys
import os
import joblib
import time
import datapreprocessing.datapreprocessing as dp
from datapreprocessing.datapreprocessing import DataCleaning
from datapreprocessing.datapreprocessing import LemmaTokenizer
from evaluation.evaluationmetrics import precision_score_plot, confusion_matrix_plot
from dataloader.dataload import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import  Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# Start time
start_time = time.time()
main_path = os.getcwd()
#Loading of data
# Define the file paths
file_paths = [
    main_path + r'\data_set\twitter_parsed_dataset.csv',
    main_path + r'\data_set\aggression_parsed_dataset.csv',
    main_path + r'\data_set\attack_parsed_dataset.csv',
    main_path + r'\data_set\kaggle_parsed_dataset.csv',
    main_path + r'\data_set\toxicity_parsed_dataset.csv',
    main_path + r'\data_set\twitter_racism_parsed_dataset.csv',
    main_path + r'\data_set\twitter_sexism_parsed_dataset.csv',
    main_path + r'\data_set\youtube_parsed_dataset.csv',
    main_path + r'\data_set\in-game data generated with ai 1.csv',
    main_path + r'\data_set\in-game data generated with ai 2.csv',
    main_path + r'\data_set\in-game data generated with ai 3.csv',
    main_path + r'\data_set\in-game data generated with ai 4.csv',
    main_path + r'\data_set\in-game data generated with ai 6.csv',  
    main_path + r'\data_set\in-game data generated with ai 5.csv',    
]

data = load_dataset(file_paths)

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(data['Text'], data['Label'], test_size=0.01, random_state=42)
print("Data split completed")

# Define the pipeline with TF-IDF and RandomForest
print("Setting up the pipeline...")
# Define the pipeline with TF-IDF and RandomForest
text_clf = Pipeline(steps=[
    ('clean', DataCleaning()),
    ('vect', TfidfVectorizer(analyzer="word", tokenizer=LemmaTokenizer(), ngram_range=(1,3), min_df=10, max_features=10000)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))  # Ensure this is RandomForest
])
print("Pipeline setup completed")

# Train the text classifier
print("Training the text classifier...")
train_start_time = time.time()
text_clf.fit(x_train, y_train)
print(f"Training completed in {time.time() - train_start_time:.2f} seconds")

# Generate predictions on the test data
print("Generating predictions on test data...")
y_predict = text_clf.predict(x_test)
y_score = text_clf.predict_proba(x_test)[:, 1]
print("Predictions generated")

# Evaluation of the model
print("Evaluating the model...")
print("Precision Score on test dataset for Random Forest: %s" % precision_score(y_test, y_predict, average='micro'))
print("AUC Score on test dataset for Random Forest: %s" % roc_auc_score(y_test, y_score, multi_class='ovo', average='macro'))
print("F1 Score on test dataset for Random Forest: %s" % f1_score(y_test, y_predict, average="weighted"))

# Plot confusion matrix
print("Plotting confusion matrix...")
confusion_matrix_plot(y_test, y_predict)
print("Confusion matrix plotted")

if text_clf is None:
    raise ValueError("The model (text_clf) is not initialized or trained properly.")

print("Storing the trained model...")

# Construct the path using os.path.join
model_dir = os.path.join(os.getcwd(), 'model')
os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

# Construct the file path
model_path = os.path.join(model_dir, 'class.pkl')
print(model_path)
# Store the model
joblib.dump(text_clf, model_path, compress=True)

# Check if the file has been created and is not empty
if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    print(f"Model successfully stored at {model_path}")
else:
    raise IOError(f"Failed to store the model at {model_path}")

print(f"Total execution time: {time.time() - start_time:.2f} seconds")