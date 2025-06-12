import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
from PIL import Image

# Add custom module path
sys.path.append(os.path.abspath(r'D:\Guvi_Project\Personalized Learning Assistant\src'))

# Import custom modules with error handling
try:
    from Data_Preprocessing import tokenization
    from Model_Training import pickle_load
    from summary_generator import summarize_text
except ImportError as e:
    st.error(f"Module import failed: {e}")
    raise

# TensorFlow & Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

st.set_page_config(
    page_title="Personalized Learning Assistant",
    page_icon=":books:",
    layout="wide"
)

st.title("AI-Powered Personalized Learning Assistant for Students")
st.header("Unlock the Future of Power Prediction with AI")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Predict Student Pass/Fail", "Score Range Prediction", "Learning Style Clustering",
    "Dropout Risk Detection", "Topic Detection from Student Answers",
    "Handwritten Digit Recognition", "AI-Based Topic Summarizer"
])

# Helper: Safe load function-----------------------------------------------------------------------
def safe_load_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"CSV file not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def safe_pickle_load(name):
    try:
        return pickle_load(name)
    except Exception as e:
        st.error(f"Error loading model component '{name}': {e}")
        return None

def safe_model_load(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Error loading Keras model from {path}: {e}")
        return None

# Section 1: Pass/Fail Prediction------------------------------------------------------------------
@st.cache_resource
def load_models():
    vectorizer = safe_pickle_load('TfidfVectorizer-sec1')
    classifier = safe_pickle_load('RandomForestClassifier-sec1')
    return vectorizer, classifier

@st.cache_data
def load_data():
    return safe_load_csv(r"D:\Guvi_Project\Personalized Learning Assistant\data\raw\ASSISTments 2009-2010 Dataset .csv")

# Section 2: Score Range Prediction----------------------------------------------------------------
@st.cache_data
def load_data_for_score_range():
    return safe_load_csv(r"D:\Guvi_Project\Personalized Learning Assistant\data\raw\student_performance_data.csv")

@st.cache_resource
def load_model_score_prediction():
    linear = safe_pickle_load('LinearRegression-sec2')
    le = safe_pickle_load('LabelEncoder-sec2')
    stdscalar = safe_pickle_load('StandardScaler-sec2')
    return linear, le, stdscalar

# Section 3: Clustering--------------------------------------------------------------------
@st.cache_data
def load_data_for_clustering():
    return safe_load_csv(r"D:\Guvi_Project\Personalized Learning Assistant\data\raw\Clustering_student_data.csv")

@st.cache_resource
def load_model_for_clustering():
    scalar = safe_pickle_load('StandardScaler-sec3')    
    kmeans = safe_pickle_load('KMeans-sec3')
    return kmeans, scalar

# Section 4: Dropout Detection--------------------------------------------------------------
@st.cache_data
def load_data_for_dropout():
    return safe_load_csv(r"D:\Guvi_Project\Personalized Learning Assistant\data\raw\student_engagement_data.csv")

@st.cache_resource
def load_model_for_dropout():
    dropout_model = safe_pickle_load('XGBClassifier-sec4')
    scalar = safe_pickle_load('StandardScaler-sec4')
    return dropout_model, scalar

# Section 5: LSTM Model for Topic Detection--------------------------------------------------
@st.cache_resource
def load_lstm_model():
    lstm_model = safe_model_load(r"D:\Guvi_Project\Personalized Learning Assistant\models\llm_models\ag_news_lstm_model.h5")
    tokenizer = safe_pickle_load('Tokenizer-sec5')
    le = safe_pickle_load('LabelEncoder-sec5')
    return lstm_model, tokenizer, le

# Section 6: CNN Model for Digit Recognition--------------------------------------------------
@st.cache_resource
def load_cnn_model():
    return safe_model_load(r"D:\Guvi_Project\Personalized Learning Assistant\models\llm_models\cnn_model.h5")

# Section 7: Summarizer-----------------------------------------------------------------------
@st.cache_resource
def summarize_model(text):
    try:
        return summarize_text(text)
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return "Unable to summarize the input text."
# Main Application Logic ---------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("Predict Student Pass/Fail")
    st.write("This model predicts whether a student will pass or fail based on their academic performance and other relevant features.")
    
    raw_df = load_data()
    if raw_df is None:
        st.warning("Dataset could not be loaded. Please check the data file.")
    else:
        try:
            df = raw_df[['skill', 'timeTaken', 'correct', 'hintCount', 'attemptCount']].copy()
            df.rename(columns={
                'skill': 'skill_id',
                'timeTaken': 'time_taken',
                'correct': 'is_correct',
                'hintCount': 'hint_count',
                'attemptCount': 'attempt_count'
            }, inplace=True)

            skillID = st.multiselect("Select Skill ID", df['skill_id'].dropna().unique(), placeholder='Select one or more skills')
            timeTaken = st.number_input("Enter Time Taken (in seconds)", 0, int(df['time_taken'].max()), 0)
            hintcount = st.number_input("Enter Hint Count", 0, int(df['hint_count'].max()), 0)
            attemptCount = st.number_input("Enter Attempt Count", 0, int(df['attempt_count'].max()), 0)

            if st.button("Predict Pass/Fail", type="primary"):
                if not skillID:
                    st.warning("Please select at least one Skill ID.")
                else:
                    skill_text = " ".join(skillID)  # Combine selected skills
                    input_data = pd.DataFrame({
                        'skill_id': [skill_text],
                        'time_taken': [timeTaken],
                        'hint_count': [hintcount],
                        'attempt_count': [attemptCount]
                    })

                    # Tokenize skill text
                    try:
                        input_data['skill_id'] = input_data['skill_id'].apply(lambda x: " ".join(tokenization(x)))
                    except Exception as e:
                        st.error(f"Tokenization failed: {e}")
                        st.stop()

                    # Load models
                    vector_model, model = load_models()
                    if vector_model is None or model is None:
                        st.error("Failed to load model or vectorizer. Please check the model files.")
                        st.stop()

                    # Vectorize and combine with other inputs
                    try:
                        vector_input_data = vector_model.transform(input_data['skill_id'])
                        combined_data = pd.concat([
                            input_data.drop(columns=['skill_id']).reset_index(drop=True),
                            pd.DataFrame(vector_input_data.toarray(), columns=vector_model.get_feature_names_out())
                        ], axis=1)
                    except Exception as e:
                        st.error(f"Vectorization or data preparation failed: {e}")
                        st.stop()

                    # Predict and display results
                    try:
                        prediction = model.predict(combined_data)
                        proba = model.predict_proba(combined_data)[0][1]

                        st.success(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        except KeyError as e:
            st.error(f"Data format error: Missing expected column {e}")

with tab2:
    st.subheader("Score Range Prediction")
    st.write("This model predicts the score range of a student based on their academic performance and other relevant features.")

    data = load_data_for_score_range()

    if data is None:
        st.warning("Dataset could not be loaded. Please check the data file.")
    else:
        try:
            time_spent = st.number_input("Enter Time Spent (in seconds)", 0, int(data['time_spent'].max()), 0)
            difficulty = st.selectbox("Select Difficulty Level", data['difficulty'].dropna().unique())
            prev_score = st.number_input("Enter Previous Score", 0, int(data['prev_score'].max()), 0)
            correct = st.selectbox("Select Correctness", data['correct'].dropna().unique())
            tag = st.selectbox("Select Tag", data['tag'].dropna().unique())

            if st.button("Predict Score Range", type="primary"):
                # Load model and preprocessors
                model, encoders, stdscalar = load_model_score_prediction()
                if model is None or encoders is None or stdscalar is None:
                    st.error("Model or preprocessing components failed to load.")
                    st.stop()

                # Prepare input data
                stu_data = pd.DataFrame({
                    'time_spent': [time_spent],
                    'difficulty': [difficulty],
                    'prev_score': [prev_score],
                    'correct': [correct],
                    'tag': [tag]
                })

                try:
                    # Apply label encoding
                    stu_data['tag'] = encoders.transform(stu_data['tag'])

                    # Standardize numerical and encoded features
                    input_scaled = stdscalar.transform(stu_data)

                    # Predict
                    prediction = model.predict(input_scaled)
                    st.success(f"🎯 **Predicted Score Range:** `{prediction[0]:.2f}`")

                except ValueError as e:
                    st.error(f"Encoding or prediction error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error during prediction: {e}")

        except KeyError as e:
            st.error(f"Dataset is missing expected column: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

with tab3:
    st.subheader("Learning Style Clustering")
    st.write("This model clusters students based on their learning styles.")

    data = load_data_for_clustering()

    if data is None:
        st.warning("Dataset could not be loaded. Please check the data file.")
    else:
        try:
            # User Inputs
            reading_efficiency = st.number_input("Reading Efficiency",0,int(data['reading_efficiency'].max()), 0)
            visual_engagement = st.number_input("Visual Engagement", 0, int(data['visual_engagement'].max()), 0)
            response_time = st.slider("Response Time (in seconds)", 0, int(data['response_time'].max()), 0)
            quiz_scores = st.slider("Quiz Scores", 0, 100, 0)

            if st.button("Cluster Learning Style", type="primary"):
                # Load model and scaler
                kmeans, scalar = load_model_for_clustering()
                if kmeans is None or scalar is None:
                    st.error("Model or scaler could not be loaded.")
                    st.stop()

                # Prepare input
                input_data = pd.DataFrame({
                    'reading_efficiency': [reading_efficiency],
                    'visual_engagement': [visual_engagement],
                    'response_time': [response_time],
                    'quiz_score': [quiz_scores]
                })

                try:
                    # Scale input data
                    input_scaled = scalar.transform(input_data)

                    # Predict cluster
                    cluster = kmeans.predict(input_scaled)
                    cluster_label = cluster[0]

                    # Interpret cluster
                    learning_style = {
                        0: 'Fast Responders',
                        1: 'Visual Learners',
                        2: 'Slow Learners'
                    }.get(cluster_label, "Unknown Cluster")

                    st.success(f"🧠 **Predicted Cluster:** `{learning_style}`")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        except KeyError as e:
            st.error(f"Data format issue: missing column {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        
with tab4:
    st.subheader("Dropout Risk Detection")
    st.write("This model predicts the risk of a student dropping out based on their academic performance and other relevant features.")

    dropout_df = load_data_for_dropout()

    if dropout_df is None:
        st.warning("Dataset could not be loaded. Please check the file path.")
    else:
        try:
            # User inputs with proper bounds
            last_login_days = st.number_input("Enter Last Login Days", 0, int(dropout_df['last_login_days'].max()), 0)
            avg_score = st.number_input("Enter Average Score", 0, int(dropout_df['avg_score'].max()), 0)
            quiz_completion_rate = st.number_input("Enter Quiz Completion Rate", 0.0, 1.0, 0.0)
            forum_posts = st.number_input("Enter Forum Posts", 0, int(dropout_df['forum_posts'].max()), 0)
            video_views_per_week = st.number_input("Enter Video Views per Week", 0, int(dropout_df['video_views_per_week'].max()), 0)
            live_session_attendance = st.number_input("Enter Live Session Attendance", 0, int(dropout_df['live_session_attendance'].max()), 0)

            if st.button("Predict Dropout Risk", type="primary"):
                # Load model and scaler
                dropout_model, scalar = load_model_for_dropout()
                if dropout_model is None or scalar is None:
                    st.error("Model or scaler failed to load. Please check the files.")
                    st.stop()

                # Create input DataFrame
                input_data = pd.DataFrame({
                    'last_login_days': [last_login_days],
                    'avg_score': [avg_score],
                    'quiz_completion_rate': [quiz_completion_rate],
                    'forum_posts': [forum_posts],
                    'video_views_per_week': [video_views_per_week],
                    'live_session_attendance': [live_session_attendance]
                })

                try:
                    # Scale input
                    input_scaled = scalar.transform(input_data)

                    # Make prediction
                    prediction = dropout_model.predict(input_scaled)

                    risk_status = "High Risk of Dropout" if prediction[0] == 1 else "Low Risk of Dropout"
                    st.success(f"📉 **Prediction:** {risk_status}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        except KeyError as e:
            st.error(f"Dataset is missing expected column: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        
with tab5:
    st.subheader("Topic Detection from Student Answers")
    st.write("This model detects the topic based on the student's answer.")

    answer = st.text_area(
        "Enter the student's answer", 
        height=100, 
        placeholder="Type or paste the student's answer here..."
    )

    if st.button("Detect Topic", type="primary"):
        if not answer.strip():
            st.warning("Please enter an answer to detect the topic.")
        else:
            model, tokenizer, le = load_lstm_model()

            if model is None or tokenizer is None or le is None:
                st.error("Model or tokenizer failed to load. Please check the files.")
                st.stop()

            try:
                # Tokenize and pad the input
                sample_seq = tokenizer.texts_to_sequences([answer])
                sample_pad = pad_sequences(sample_seq, maxlen=100)

                # Predict
                pred = model.predict(sample_pad)
                pred_index = np.argmax(pred)

                # Decode label
                try:
                    predicted_label = le.inverse_transform([pred_index])[0]
                except Exception:
                    predicted_label = str(pred_index)  # Fallback if decoding fails

                # Map known labels to human-readable topics
                topic_map = {
                    '0': 'World',
                    '1': 'Sports',
                    '2': 'Business',
                    '3': 'Science and Technology',
                    0: 'World',
                    1: 'Sports',
                    2: 'Business',
                    3: 'Science and Technology',
                }

                predicted_topic = topic_map.get(predicted_label, 'Unknown')

                st.success(f"📚 **Detected Topic:** `{predicted_topic}`")

            except Exception as e:
                st.error(f"Topic detection failed: {e}")

with tab6:
    st.subheader("Handwritten Digit Recognition")
    st.write("This model recognizes handwritten digits from images.")

    uploaded_file = st.file_uploader("Upload an image of a handwritten digit", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Load and preprocess the image
            image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
            image = image.resize((28, 28))  # Resize to 28x28 pixels
            image_array = np.array(image) / 255.0  # Normalize pixel values
            image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for CNN input

            # Load CNN model
            cnn_model = load_cnn_model()
            if cnn_model is None:
                st.error("CNN model could not be loaded. Please check the model file.")
                st.stop()

            # Make prediction
            prediction = cnn_model.predict(image_array)
            digit = np.argmax(prediction)

            st.success(f"🔢 **Predicted Digit:** `{digit}`")
            st.image(uploaded_file, caption='Uploaded Image')

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Please upload an image to begin digit recognition.")

with tab7:
    st.subheader("AI-Based Topic Summarizer")
    st.write("This model summarizes the topic based on the student's answer.")

    text = st.text_area("Enter the text to summarize", height=100, placeholder="Type or paste your text here...")

    if st.button("Summarize", type="primary"):
        if not text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            try:
                summary = summarize_model(text)
                if not summary or summary.strip() == "":
                    st.warning("The summarizer did not return any content. Try with more informative input.")
                else:
                    st.success("📝 **Summary:**")
                    st.write(summary)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
