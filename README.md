AI-Powered Personalized Learning Assistant
**Project Overview**
This project aims to build an AI-powered Personalized Learning Assistant that provides:

*Student performance prediction
*Score/grade regression
*Dropout detection
*Learning style clustering
*Handwritten digit recognition
*Text summarization of study material

It utilizes classical machine learning, deep learning, and NLP techniques to enhance personalized education through analytics and intelligent predictions.

**Dataset Used**
**Sources**
*Multiple open-source datasets were used in this project:
*EdNet, ASSISTments, and Khan Academy logs for student interaction and performance.
*MNIST for handwritten digit recognition.
*TED Talks and custom educational articles for summarization tasks.

**Preprocessing Steps**
*Handling missing values using imputation or removal.
*Outlier detection and treatment using Winsorization.
*Label encoding for categorical variables.
*Text cleaning (stopword removal, stemming, tokenization).
*Sequence padding and tokenization for NLP models.
*Normalization and oversampling to balance data.

**Machine Learning Techniques Applied**
Task	Technique
Student Score Prediction	-  Random Forest Regressor, Linear Regression
Dropout Detection	-  XGBoost Classifier, Logistic Regression
Text Summarization	-  Transformer-based summarization using Hugging Face
Learning Style Clustering	-  KMeans, PCA
Digit Recognition	-  CNN with Keras and TensorFlow
NLP Tokenization & Embedding	-  TF-IDF, Tokenizer, LSTM Embedding

The project incorporates hyperparameter tuning, model evaluation metrics, and visualizations for performance comparison.

**Deployment Instructions**
Requirements
Ensure the following packages are installed (via requirements.txt or manually):

pip install -r requirements.txt

Key libraries:

*streamlit
*tensorflow
*scikit-learn
*transformers
*nltk
*seaborn, matplotlib, missingno

Running the App

streamlit run app.py

The app has separate tabs for data upload, EDA, preprocessing, model training, evaluation, and prediction.

Ensure all necessary .pkl, encoder, and tokenizer files are in the correct paths as specified in the codebase.

