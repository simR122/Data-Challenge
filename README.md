# AUTO-REQ-video-pre-requisite-data-NLP

## Overview
This project focuses on identifying prerequisite relationships between academic videos to enhance the instructional flow on online learning platforms. Using advanced Natural Language Processing (NLP) techniques, it preprocesses and analyzes video transcripts to facilitate a more cohesive learning experience.

## Notebook
The main notebook for this project is accessible on Google Colab. You can view it [here](https://colab.research.google.com/drive/1_D00OOSqpFrAKUIXPr4DHSEu_x0NuCzl?usp=sharing).

## Preprocessing Techniques
The following techniques were applied to prepare the video transcripts:

- **Cleaning**: Removed symbols, special characters, and numbers.
- **Tokenization**: Split text into individual words.
- **Stopword Removal**: Filtered out common words that do not contribute meaning.
- **Lemmatization**: Converted words to their base or root form.

## Algorithm
The model architecture uses contextual embeddings from BERT in combination with a Bidirectional GRU (Gated Recurrent Unit) layer to capture sequential dependencies. This approach enables binary classification of the prerequisite relationships between videos. Adjusting hyperparameters and exploring additional pre-trained BERT models could further optimize performance.

## Approach
1. **Data Preprocessing**: Applied cleaning, tokenization, stopword removal, and lemmatization for structured text data.
2. **Class Balancing**: Addressed class imbalance by downsampling the majority class.
3. **Data Integration**: Merged processed data with the original dataset based on prerequisite information.
4. **Model Exploration**: Evaluated a range of models to identify the best performance:
   - **Supervised ML Models**: SVM, Logistic Regression, Random Forest.
   - **Unsupervised Learning**: K-Means and hierarchical clustering.
   - **Graph Neural Networks & Ensemble Methods**: Gradient Boosting and Stacking.
5. **Text Classification**: Used BERT + Bi-GRU for binary classification, achieving optimal F1 score and accuracy.

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

