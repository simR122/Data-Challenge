# AUTO-REQ-video-pre-requisite-data-NLP

## Overview
This repository contains the code for the AUTO-REQ-video-pre-requisite-data project, which is focused on identifying pre-requisites between academic videos to enhance the learning experience on online platforms. The project uses Natural Language Processing (NLP) techniques to preprocess and analyze the video transcripts.

## Notebook
The main notebook for this project is located on Google Colab. You can access it [here](https://colab.research.google.com/drive/1_D00OOSqpFrAKUIXPr4DHSEu_x0NuCzl?usp=sharing).

## Preprocessing Techniques Used
The following preprocessing techniques were used on the video transcripts:

- **Cleaning**: Removal of symbols, special characters, and numbers.
- **Tokenization**: Splitting text into individual words.
- **Stopword Removal**: Removal of common words that do not contribute to the meaning of the text.
- **Lemmatization**: Converting words to their base or root form.

## Algorithm Used
The algorithm used in this project leverages the contextual embeddings from BERT and captures sequential information using the Bidirectional GRU (Gated Recurrent Unit) layer. The model is trained to classify text data into binary categories. The training loop updates the model parameters to minimize the loss, and the evaluation provides insights into the model's performance on the test set. Adjusting hyperparameters and experimenting with different pre-trained BERT models could further optimize the model's performance.

## Approach
1. **Data Preprocessing**: Applied techniques such as cleaning, tokenization, stopword removal, and lemmatization to prepare the video transcripts.
2. **Downsampling**: Balanced the dataset by downsampling the majority class.
3. **Data Merging**: Integrated the preprocessed data with the original dataset based on pre-requisite information.
4. **Model Exploration**: Experimented with various models to find the best performance, including:
   - **Supervised ML Models**: SVM, Logistic Regression, and Random Forests.
   - **Unsupervised Learning**: K-Means and hierarchical clustering.
   - **Graph Neural Networks** and **Ensemble Methods**: Gradient Boosting and Stacking.
5. **Text Classification**: Used BERT + Bi-GRU for binary classification, achieving the highest F1 score and accuracy.


## Usage

### 1. Request Dataset Access
- Request access to the dataset from the challenge website.

### 2. Preprocess Data
- Download the dataset and place it in a suitable directory.
- Update the dataset path in the `preprocessing.py` file.
- Run the `preprocessing.py` script to preprocess the data:
  ```bash
  python preprocessing.py

### 2. Preprocess Data
- Ensure the preprocessed data is correctly saved and available.
- Update the dataset path in the train.py file if necessary.
- Run the train.py script to train the model:
  ```bash
  python train.py

