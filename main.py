import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import streamlit as st
import joblib
import pandas as pd
import re
import os

# Title of the app
st.title("AI vs Human Text Detect")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

# Function to load models
def load_models():
    try:
        clf_rf = joblib.load('joblib_files/random_forest_model.joblib')
        # clf_knn = joblib.load('joblib_files/knn_model.joblib')
        clf_mlp = joblib.load('joblib_files/multilayer_perceptron_model.joblib')
        clf_gb = joblib.load('joblib_files/gradient_boosting_model.joblib')
        clf_nb = joblib.load('joblib_files/naive_bayes_model.joblib')
        clf_xgb = joblib.load('joblib_files/xgboost_model.joblib')
        clf_dt = joblib.load('joblib_files/decision_tree_model.joblib')
        clf_lgbm = joblib.load('joblib_files/lightgbm_model.joblib')
        clf_svm = joblib.load('joblib_files/svm_model.joblib')
        tfidf = joblib.load('joblib_files/tfidf_vectorizer.joblib')
        
        return clf_rf, clf_knn, clf_mlp, clf_gb, clf_nb, clf_xgb, clf_dt, clf_lgbm, clf_svm, tfidf
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models
models = load_models()

# Input form
st.header("Enter your text below:")

# Custom CSS to make the text area larger
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 300px;
        width: 200%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

message = st.text_area("Message")

# Minimum word count
min_word_count = 1

# CSV file path to store user feedback
feedback_csv = "feedback.csv"

if st.button("Submit"):
    # Check if the message has at least the minimum word count
    word_count = len(message.split())

    if word_count < min_word_count:
        st.warning(f'Entered text must be a minimum of {min_word_count} words. You have entered {word_count} words.')
    else:
        # Preprocess the input message
        preprocessed_message = preprocess_text(message)

        # Transform the preprocessed message using the TF-IDF vectorizer
        try:
            text_features = models[9].transform([preprocessed_message])
        except Exception as e:
            st.error(f"Error in transforming text: {e}")
            st.stop()

        # Predict using the models
        predictions = {}
        try:
            predictions['Random Forest'] = models[0].predict(text_features)[0]
            # predictions['KNN'] = models[1].predict(text_features)[0]
            predictions['Multilayer Perceptron'] = models[2].predict(text_features)[0]
            predictions['Gradient Boosting'] = models[3].predict(text_features)[0]
            predictions['Naive Bayes'] = models[4].predict(text_features)[0]
            predictions['XGBoost'] = models[5].predict(text_features)[0]
            predictions['Decision Tree'] = models[6].predict(text_features)[0]
            predictions['LightGBM'] = models[7].predict(text_features)[0]
            predictions['SVM'] = models[8].predict(text_features.toarray())[0]
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Aggregating results or showing individual model predictions
        ai_votes = sum(1 for v in predictions.values() if v == 1)
        human_votes = sum(1 for v in predictions.values() if v == 0)

        # Determine final prediction based on majority voting
        if ai_votes > human_votes:
            prediction_result = 'AI'
            st.session_state.results = (preprocessed_message, prediction_result, predictions)
            st.success('The text is likely written by AI')
        elif human_votes > ai_votes:
            prediction_result = 'Human'
            st.session_state.results = (preprocessed_message, prediction_result, predictions)
            st.success('The text is likely written by a Human')
        else:
            prediction_result = 'Inconclusive'
            st.session_state.results = (preprocessed_message, prediction_result, predictions)
            st.warning('The models are inconclusive; the text might be either AI or Human generated.')


        # # Display individual model predictions for transparency
        # st.subheader("Model Predictions:")
        # for model_name, prediction in predictions.items():
        #     st.write(f"{model_name}: {'AI' if prediction == 1 else 'Human'}")


if st.session_state.results:
    preprocessed_message, prediction_result, predictions = st.session_state.results

    # Feedback form
    st.subheader("Is the prediction correct?")

    # Create two buttons for feedback
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes"):
            feedback_data = pd.DataFrame({
                'Text': [preprocessed_message],
                'Prediction': [prediction_result],
            })
            try:
                feedback_data.to_csv(feedback_csv, mode='a', header=not os.path.exists(feedback_csv), index=False)
                st.success("Thank you for your feedback! Your data has been saved.")
                st.session_state.feedback_submitted = True
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

    with col2:
        if st.button("No"):
            # Check the prediction result to determine the correct label
            if prediction_result == 'AI':
                correct_prediction = 'Human'
            else:  # prediction_result == 'Human'
                correct_prediction = 'AI'

            feedback_data = pd.DataFrame({
                'Text': [preprocessed_message],
                'Prediction': [correct_prediction],
            })
            try:
                feedback_data.to_csv(feedback_csv, mode='a', header=not os.path.exists(feedback_csv), index=False)
                st.success("Thank you for your feedback! Your data has been saved.")
                st.session_state.feedback_submitted = True
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

    if st.session_state.feedback_submitted:
        st.write("You can submit another text if you wish.")
