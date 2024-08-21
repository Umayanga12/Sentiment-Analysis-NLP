# Sentiment Analysis using NLP

This repository contains a sentiment analysis model implemented in Python using Natural Language Processing (NLP) techniques. The model is built in a Jupyter notebook and demonstrates various steps such as data preprocessing, vectorization, and classification of text data into positive or negative sentiment.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Workflow](#model-workflow)
- [Results](#results)
- [License](#license)

## Introduction

This project aims to build a sentiment analysis model using Python and NLP techniques. The model is designed to classify text data into positive or negative sentiment categories. The Jupyter notebook provided in this repository contains the step-by-step implementation of the model.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Umayanga12/Sentiment-Analysis-NLP.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Sentiment-Analysis-NLP
   ```
3. Install the necessary packages:
   ```bash
   pip install pandas numpy scikit-learn nltk
   ```

4. Download NLTK stopwords (if not already installed):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage

To run the sentiment analysis model, open the Jupyter notebook `Sentiment_Analysis_using_NLP.ipynb` in your local environment:

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Sentiment_Analysis_using_NLP.ipynb` and run the cells sequentially.

The notebook will guide you through the process of text preprocessing, vectorization using TF-IDF, and building the sentiment analysis model.

## Project Structure

- `Sentiment_Analysis_using_NLP.ipynb`: The main Jupyter notebook containing the implementation of the sentiment analysis model.

## Model Workflow

1. **Data Preprocessing**:
   - Convert text to lowercase.
   - Remove punctuation.
   - Remove stopwords.
   - Apply stemming using `PorterStemmer`.

2. **Feature Extraction**:
   - Convert the preprocessed text into numerical features using `TfidfVectorizer`.

3. **Model Training**:
   - Train a machine learning model (e.g., Logistic Regression, SVM) using the TF-IDF features to classify sentiment.

4. **Evaluation**:
   - Assess the performance of the model using metrics such as accuracy, precision, recall, and F1-score.

## Results

The results of the model are displayed in the notebook, including visualizations and performance metrics. The model successfully classifies text into positive or negative sentiment with high accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
