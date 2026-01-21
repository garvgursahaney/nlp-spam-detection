# Spam Detection with Natural Language Processing

This project makes use of different NLP techniques to implement a text classification system that can in turn detect spam messages.

## Features
- Automatic dataset download (UCI SMS Spam Collection)
- Text cleaning, stemming, stopword removal, and lemmatization
- Models:
  - TF-IDF + Logistic Regression
  - Bag-of-Words + Logistic Regression
  - TF-IDF + Naive Bayes
- Hyperparameter tuning via GridSearchCV
- Evaluation using classification report and confusion matrix
- Demo predictions on any new incoming messages

## Installation

pip install -r requirements.txt

## Run

python main.py

## Output
- Prints evaluation metrics
- Visualizes a confusion matrix
- Shows an example of predictions
