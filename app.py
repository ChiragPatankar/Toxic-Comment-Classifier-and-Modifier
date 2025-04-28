import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pickle
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize the stemmer
stemmer = SnowballStemmer('english')
stop_words_set = set(stopwords.words('english'))


# Text preprocessing functions
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words_set])


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def stemming(sentence):
    return " ".join([stemmer.stem(word) for word in str(sentence).split()])


def preprocess_text(text):
    text = remove_stopwords(text)
    text = clean_text(text)
    text = stemming(text)
    return text


# Function to get sentiment
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive", score
    elif score < 0:
        return "Negative", score
    else:
        return "Neutral", score


# Function to moderate text based on toxicity
def moderate_text(text, predictions, threshold_moderate=0.5, threshold_delete=0.8):
    # Check if any toxicity class exceeds the delete threshold
    if any(pred >= threshold_delete for pred in predictions):
        return "*** COMMENT DELETED DUE TO HIGH TOXICITY ***", "delete"

    # Check if any toxicity class exceeds the moderate threshold
    elif any(pred >= threshold_moderate for pred in predictions):
        # List of potentially toxic words to censor
        toxic_words = ["stupid", "idiot", "dumb", "hate", "sucks", "terrible",
                       "awful", "garbage", "trash", "pathetic", "ridiculous"]

        words = text.split()
        moderated_words = []

        for word in words:
            # Clean word for comparison
            clean_word = re.sub(r'[^\w\s]', '', word.lower())

            # Check if the word is in the toxic words list
            if clean_word in toxic_words:
                # Replace with a more neutral placeholder
                moderated_words.append("[inappropriate]")
            else:
                moderated_words.append(word)

        return " ".join(moderated_words), "moderate"

    # If no toxicity is detected
    else:
        return text, "keep"


# Function to train and save the model
def train_model(X_train, y_train, model_type='logistic_regression'):
    st.write("Training model...")

    # Ensure `y_train` has 6 columns
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Create missing columns if they don't exist
    for col in label_columns:
        if col not in y_train.columns:
            y_train[col] = 0

    # Ensure columns are in the right order
    y_train = y_train[label_columns]

    if model_type == 'logistic_regression':
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=50000)),
            ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000), n_jobs=-1))
        ])
    else:  # Naive Bayes
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=50000)),
            ('clf', OneVsRestClassifier(MultinomialNB(), n_jobs=-1))
        ])

    pipeline.fit(X_train, y_train)

    return pipeline


# Function to evaluate model performance
def evaluate_model(pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)

    # Get predicted probabilities
    pred_probs = pipeline.predict_proba(X_test)

    # Handle single-label predictions
    if isinstance(pred_probs, list) and len(pred_probs) == 1:
        pred_probs = pred_probs[0]  # Get the first element if it's a list with one element

    accuracy = accuracy_score(y_test, predictions)

    # Safely calculate ROC AUC score (handle potential errors)
    try:
        roc_auc = roc_auc_score(y_test, pred_probs, average='macro')
    except Exception as e:
        st.warning(f"Could not calculate ROC AUC score: {str(e)}")
        roc_auc = 0.0

    return accuracy, roc_auc, predictions, pred_probs


# Function to create a download link for the trained model
def get_model_download_link(model, filename):
    model_bytes = pickle.dumps(model)
    b64 = base64.b64encode(model_bytes).decode()
    href = f'<a href="data:file/pickle;base64,{b64}" download="{filename}">Download Trained Model</a>'
    return href


# Function to plot toxicity distribution
def plot_toxicity_distribution(df, toxicity_columns):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df[toxicity_columns].sum()
    sns.barplot(x=x.index, y=x.values, alpha=0.8, palette='viridis', ax=ax)

    plt.title('Toxicity Distribution')
    plt.ylabel('Count')
    plt.xlabel('Toxicity Category')
    plt.xticks(rotation=45)

    return fig


# Function to provide sample data format
def show_sample_data_format():
    st.subheader("Sample Data Format")

    # Create sample dataframe
    sample_data = {
        'comment_text': [
            "This is a normal comment.",
            "This is a toxic comment you idiot!",
            "You're all worthless and should die.",
            "I respectfully disagree with your point."
        ],
        'toxic': [0, 1, 1, 0],
        'severe_toxic': [0, 0, 1, 0],
        'obscene': [0, 1, 0, 0],
        'threat': [0, 0, 1, 0],
        'insult': [0, 1, 1, 0],
        'identity_hate': [0, 0, 0, 0]
    }

    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)

    # Create download link for sample data
    csv = sample_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_toxic_data.csv">Download Sample CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.info("""
    Your CSV file should contain:
    1. A column with comment text
    2. One or more columns with binary values (0 or 1) for each toxicity category
    """)


# Function to validate dataset
def validate_dataset(df, comment_column, toxicity_columns):
    issues = []

    # Check if comment column exists
    if comment_column not in df.columns:
        issues.append(f"Comment column '{comment_column}' not found in the dataset")

    # Check if toxicity columns exist
    missing_columns = [col for col in toxicity_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing toxicity columns: {', '.join(missing_columns)}")

    # Check if values in toxicity columns are valid (0 or 1)
    for col in toxicity_columns:
        if col in df.columns:
            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' contains non-numeric values")
            else:
                # Check for values other than 0 and 1
                invalid_values = df[col].dropna().apply(lambda x: x not in [0, 1, 0.0, 1.0])
                if invalid_values.any():
                    issues.append(f"Column '{col}' contains values other than 0 and 1")

    # Check for empty data
    if df.empty:
        issues.append("Dataset is empty")
    elif df[comment_column].isna().all():
        issues.append("Comment column contains no data")

    return issues


# Function to extract predictions from model output
def extract_predictions(predictions_proba, toxicity_categories):
    """
    Helper function to extract probabilities from model output,
    handling different output formats.
    """
    # Debug information
    if st.session_state.debug_mode:
        st.write(f"Predictions type: {type(predictions_proba)}")
        st.write(
            f"Predictions shape/length: {np.shape(predictions_proba) if hasattr(predictions_proba, 'shape') else len(predictions_proba)}")

    # Case 1: List of arrays with one element per toxicity category
    if isinstance(predictions_proba, list) and len(predictions_proba) == len(toxicity_categories):
        return [pred_array[:, 1][0] if pred_array.shape[1] > 1 else pred_array[0] for pred_array in predictions_proba]

    # Case 2: List with a single array (common for OneVsRestClassifier)
    elif isinstance(predictions_proba, list) and len(predictions_proba) == 1:
        pred_array = predictions_proba[0]
        # If it's a 2D array with number of columns equal to number of categories
        if len(pred_array.shape) == 2 and pred_array.shape[1] == len(toxicity_categories):
            return pred_array[0]  # Return first row, which contains all probabilities
        # If it's a 2D array with 2 columns per category (common binary classifier output)
        elif len(pred_array.shape) == 2 and pred_array.shape[1] == 2:
            return np.array([pred_array[0, 1]])

    # Case 3: Direct numpy array
    elif isinstance(predictions_proba, np.ndarray):
        # If it's already the right shape
        if len(predictions_proba.shape) == 2 and predictions_proba.shape[1] == len(toxicity_categories):
            return predictions_proba[0]
        # If it's a 2D array with two columns (binary classification)
        elif len(predictions_proba.shape) == 2 and predictions_proba.shape[1] == 2:
            # For binary classification, return the probability of positive class
            return np.array([predictions_proba[0, 1]])

    # If prediction format isn't recognized, return a repeated array of single probability
    # This handles the case where we only have one prediction but need to repeat it
    if isinstance(predictions_proba, list) and len(predictions_proba) == 1:
        single_prob = predictions_proba[0]
        if hasattr(single_prob, 'shape') and len(single_prob.shape) == 2 and single_prob.shape[1] == 2:
            # Take positive class probability and repeat for all categories
            return np.full(len(toxicity_categories), single_prob[0, 1])

    # Last resort fallback
    st.warning(f"Unexpected prediction format. Creating default predictions.")
    return np.zeros(len(toxicity_categories))



# Streamlit app
def main():
    st.title("Toxic Comment Classifier and Moderator")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'toxicity_categories' not in st.session_state:
        st.session_state.toxicity_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    # Sidebar
    st.sidebar.header("Options")

    # Debug mode toggle
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode)

    # Reset model button
    if st.sidebar.button("Reset Model"):
        st.session_state.model = None
        st.session_state.toxicity_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        st.experimental_rerun()

    # Navigation
    page = st.sidebar.selectbox("Choose a page", ["Home", "Analyze Comments", "Batch Processing", "Train Model"])

    if page == "Home":
        st.write("""
        ## Welcome to the Toxic Comment Classifier

        This app helps you classify and moderate potentially toxic comments. You can:

        1. **Analyze individual comments** to check their toxicity levels
        2. **Process multiple comments** by uploading a CSV file
        3. **Train a new model** using your own labeled dataset

        The app uses machine learning to classify comments into different toxicity categories:

        - Toxic
        - Severe Toxic
        - Obscene
        - Threat
        - Insult
        - Identity Hate

        It also provides sentiment analysis and automatic moderation features.
        """)

        st.write("---")

        st.write("""
        ### How to use:

        1. Navigate to the **Analyze Comments** page to check individual comments
        2. Go to the **Batch Processing** page to analyze multiple comments
        3. Use the **Train Model** page to train a new model with your own data
        """)

        # Show sample data format
        if st.button("Show Sample Data Format"):
            show_sample_data_format()

    elif page == "Analyze Comments":
        st.header("Analyze Individual Comments")

        # Check if model is loaded
        if st.session_state.model is None:
            st.warning("No model is loaded. Please upload a model or train a new one.")

            # Option to load a pre-trained model
            st.subheader("Upload Pre-trained Model")
            model_file = st.file_uploader("Upload a pickle file of your trained model", type=["pkl"])

            if model_file is not None:
                try:
                    st.session_state.model = pickle.load(model_file)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

        else:
            # Set the thresholds for moderation
            st.subheader("Moderation Settings")
            col1, col2 = st.columns(2)
            with col1:
                threshold_moderate = st.slider("Threshold for moderate toxicity", 0.0, 1.0, 0.5, 0.05)
            with col2:
                threshold_delete = st.slider("Threshold for high toxicity", 0.0, 1.0, 0.8, 0.05)

            # User input
            st.subheader("Enter a comment to analyze")
            comment = st.text_area("Comment", height=100)

            if st.button("Analyze"):
                if comment:
                    # Preprocess the comment
                    processed_comment = preprocess_text(comment)

                    # Debug information
                    if st.session_state.debug_mode:
                        st.write("Processed comment:", processed_comment)
                        st.write("Model type:", type(st.session_state.model))
                        if hasattr(st.session_state.model, 'named_steps'):
                            st.write("Pipeline steps:", list(st.session_state.model.named_steps.keys()))
                            if 'clf' in st.session_state.model.named_steps:
                                st.write("Classifier type:", type(st.session_state.model.named_steps['clf']))
                                st.write("Is OneVsRest?",
                                         isinstance(st.session_state.model.named_steps['clf'], OneVsRestClassifier))

                    try:
                        # Get predictions
                        predictions_proba = st.session_state.model.predict_proba([processed_comment])

                        # Debug information
                        if st.session_state.debug_mode:
                            st.write("Raw predictions type:", type(predictions_proba))
                            st.write("Raw predictions shape:", len(predictions_proba))
                            if isinstance(predictions_proba, list):
                                st.write("First prediction element type:", type(predictions_proba[0]))
                                if hasattr(predictions_proba[0], 'shape'):
                                    st.write("First prediction shape:", predictions_proba[0].shape)

                        # Extract probabilities using the helper function
                        probabilities = extract_predictions(predictions_proba, st.session_state.toxicity_categories)

                        # Debug information
                        if st.session_state.debug_mode:
                            st.write("Extracted probabilities:", probabilities)
                            st.write("Probabilities length:", len(probabilities))

                        # Check if we have the correct number of probabilities
                        if len(probabilities) != len(st.session_state.toxicity_categories):
                            st.error(
                                f"Model prediction mismatch! Expected {len(st.session_state.toxicity_categories)} categories but got {len(probabilities)}.")
                            if st.session_state.debug_mode:
                                st.write("Model toxicity categories:", st.session_state.toxicity_categories)
                                st.write("Prediction shape:", len(probabilities))
                        else:
                            # Display results
                            st.subheader("Analysis Results")

                            # Create a DataFrame for the results
                            results_df = pd.DataFrame({
                                'Category': st.session_state.toxicity_categories,
                                'Probability': probabilities
                            })

                            # Display the probabilities
                            st.write("Toxicity Probabilities:")

                            # Create a bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Category', y='Probability', data=results_df, palette='viridis', ax=ax)
                            plt.title('Toxicity Probabilities')
                            plt.ylabel('Probability')
                            plt.xlabel('Category')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)

                            # Show the table
                            st.dataframe(results_df)

                            # Moderate the comment
                            moderated_comment, action = moderate_text(comment, probabilities, threshold_moderate,
                                                                      threshold_delete)

                            # Display the moderation result
                            st.subheader("Moderation Result")

                            if action == "delete":
                                st.error(moderated_comment)
                            elif action == "moderate":
                                st.warning(f"Moderated Comment: {moderated_comment}")
                            else:
                                st.success(f"Original Comment (Passed): {moderated_comment}")

                            # Sentiment analysis
                            sentiment, score = get_sentiment(comment)
                            st.subheader("Sentiment Analysis")

                            # Display sentiment with color coding
                            if sentiment == "Positive":
                                st.success(f"Sentiment: {sentiment} (Score: {score:.2f})")
                            elif sentiment == "Negative":
                                st.error(f"Sentiment: {sentiment} (Score: {score:.2f})")
                            else:
                                st.info(f"Sentiment: {sentiment} (Score: {score:.2f})")
                    except Exception as e:
                        st.error(f"Error analyzing comment: {str(e)}")
                        if st.session_state.debug_mode:
                            st.write("Debug information:")
                            import traceback
                            st.write("Traceback:", traceback.format_exc())
                else:
                    st.warning("Please enter a comment to analyze.")

    elif page == "Batch Processing":
        st.header("Batch Processing")

        # Check if model is loaded
        if st.session_state.model is None:
            st.warning("No model is loaded. Please upload a model or train a new one.")

            # Option to load a pre-trained model
            st.subheader("Upload Pre-trained Model")
            model_file = st.file_uploader("Upload a pickle file of your trained model", type=["pkl"])

            if model_file is not None:
                try:
                    st.session_state.model = pickle.load(model_file)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

        else:
            # Upload CSV file
            st.subheader("Upload CSV with Comments")
            csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

            if csv_file is not None:
                # Read the CSV file
                try:
                    df = pd.read_csv(csv_file)

                    # Show preview
                    st.write("Preview of the data:")
                    st.dataframe(df.head())

                    # Select the comment column
                    st.write("Select the column containing comments:")
                    comment_column = st.selectbox("Comment Column", df.columns)

                    # Set the thresholds for moderation
                    st.subheader("Moderation Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        threshold_moderate = st.slider("Threshold for moderate toxicity", 0.0, 1.0, 0.5, 0.05)
                    with col2:
                        threshold_delete = st.slider("Threshold for high toxicity", 0.0, 1.0, 0.8, 0.05)

                    if st.button("Process Comments"):
                        # Create a new DataFrame for results
                        results_df = df.copy()

                        # Add columns for toxicity probabilities
                        for category in st.session_state.toxicity_categories:
                            results_df[f'prob_{category}'] = 0.0

                        # Add columns for moderation and sentiment
                        results_df['moderated_comment'] = ""
                        results_df['moderation_action'] = ""
                        results_df['sentiment'] = ""
                        results_df['sentiment_score'] = 0.0

                        # Show progress bar
                        progress_bar = st.progress(0)

                        # Count successful and failed analyses
                        success_count = 0
                        error_count = 0

                        # Process each comment
                        for i, row in df.iterrows():
                            # Update progress
                            progress_bar.progress((i + 1) / len(df))

                            # Get the comment
                            comment = row[comment_column]

                            if pd.isna(comment) or comment == "":
                                continue

                            try:
                                # Preprocess the comment
                                processed_comment = preprocess_text(comment)

                                # Get predictions
                                predictions_proba = st.session_state.model.predict_proba([processed_comment])

                                # Extract probabilities
                                probabilities = extract_predictions(predictions_proba,
                                                                    st.session_state.toxicity_categories)

                                # Store the probabilities
                                for j, category in enumerate(st.session_state.toxicity_categories):
                                    if j < len(probabilities):
                                        results_df.at[i, f'prob_{category}'] = probabilities[j]

                                # Moderate the comment
                                moderated_comment, action = moderate_text(
                                    comment,
                                    probabilities,
                                    threshold_moderate,
                                    threshold_delete
                                )
                                results_df.at[i, 'moderated_comment'] = moderated_comment
                                results_df.at[i, 'moderation_action'] = action

                                # Get sentiment
                                sentiment, score = get_sentiment(comment)
                                results_df.at[i, 'sentiment'] = sentiment
                                results_df.at[i, 'sentiment_score'] = score

                                success_count += 1
                            except Exception as e:
                                error_count += 1
                                if st.session_state.debug_mode:
                                    st.error(f"Error processing comment at row {i}: {str(e)}")

                        # Display the results
                        st.subheader("Processing Results")
                        st.success(f"Successfully processed {success_count} comments")

                        if error_count > 0:
                            st.warning(f"Failed to process {error_count} comments")

                        st.dataframe(results_df)

                        # Visualize toxicity distribution
                        st.subheader("Toxicity Distribution")

                        # Create a summary of toxicity probabilities
                        toxicity_summary = pd.DataFrame({
                            'Category': st.session_state.toxicity_categories,
                            'Average Probability': [results_df[f'prob_{category}'].mean() for category in
                                                    st.session_state.toxicity_categories]
                        })

                        # Create a bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Category', y='Average Probability', data=toxicity_summary, palette='viridis',
                                    ax=ax)
                        plt.title('Average Toxicity Probabilities')
                        plt.ylabel('Average Probability')
                        plt.xlabel('Category')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                        # Visualize moderation actions
                        st.subheader("Moderation Actions")

                        # Count moderation actions
                        moderation_counts = results_df['moderation_action'].value_counts()

                        # Create a pie chart
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(moderation_counts, labels=moderation_counts.index, autopct='%1.1f%%', startangle=90,
                               colors=['green', 'orange', 'red'])
                        ax.axis('equal')
                        plt.title('Moderation Actions')
                        st.pyplot(fig)

                        # Visualize sentiment distribution
                        st.subheader("Sentiment Distribution")

                        # Count sentiment values
                        sentiment_counts = results_df['sentiment'].value_counts()

                        # Create a pie chart
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
                               colors=['green', 'blue', 'red'])
                        ax.axis('equal')
                        plt.title('Sentiment Distribution')
                        st.pyplot(fig)

                        # Create a download link for the results
                        csv = results_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="moderated_comments.csv">Download Results as CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    if st.session_state.debug_mode:
                        import traceback
                        st.write("Traceback:", traceback.format_exc())
            else:
                st.info("Please upload a CSV file containing comments to process.")

    elif page == "Train Model":
        st.header("Train New Model")

        # Upload training data
        st.subheader("Upload Training Data")
        st.info(
            "The training data should be a CSV file with a column for comments and columns for toxicity labels (0 or 1).")

        # Show sample data format button
        if st.button("Show Sample Data Format"):
            show_sample_data_format()

        training_file = st.file_uploader("Upload a CSV file with labeled data", type=["csv"])

        if training_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(training_file)

                # Show the first few rows
                st.write("Preview of the data:")
                st.dataframe(df.head())

                # Select the comment column
                st.write("Select the column containing comments:")
                comment_column = st.selectbox("Comment Column", df.columns)

                # Select the toxicity columns
                st.write("Select the toxicity label columns:")
                toxicity_columns = st.multiselect("Toxicity Columns", df.columns.tolist(),
                                                  default=[col for col in df.columns if
                                                           col != comment_column and col in st.session_state.toxicity_categories])

                if not toxicity_columns:
                    st.warning("Please select at least one toxicity column.")
                else:
                    # Validate the dataset
                    issues = validate_dataset(df, comment_column, toxicity_columns)

                    if issues:
                        st.error("Data validation issues:")
                        for issue in issues:
                            st.warning(issue)

                        # Show detailed information in debug mode
                        if st.session_state.debug_mode:
                            st.subheader("Debug Information")
                            for col in toxicity_columns:
                                if col in df.columns:
                                    st.write(f"Column '{col}' unique values: {df[col].unique()}")
                                    st.write(f"Column '{col}' data type: {df[col].dtype}")
                    else:
                        # Select the model type
                        model_type = st.selectbox("Select Model Type", ["logistic_regression", "naive_bayes"])

                        # Split ratio
                        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

                        if st.button("Train Model"):
                            # Preprocess the comments
                            with st.spinner("Preprocessing comments..."):
                                st.write("Preprocessing comments...")
                                df['processed_comment'] = df[comment_column].apply(preprocess_text)

                            # Split the data
                            X = df['processed_comment']
                            y = df[toxicity_columns]

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)

                            # Train the model
                            with st.spinner("Training model..."):
                                model = train_model(X_train, y_train, model_type)

                            if model is not None:
                                # Debug information
                                if st.session_state.debug_mode:
                                    st.write("Model type:", type(model))
                                    st.write("Pipeline steps:", list(model.named_steps.keys()))
                                    st.write("Classifier type:", type(model.named_steps['clf']))
                                    st.write("Is OneVsRest?", isinstance(model.named_steps['clf'], OneVsRestClassifier))

                                # Evaluate the model
                                with st.spinner("Evaluating model..."):
                                    accuracy, roc_auc, predictions, pred_probs = evaluate_model(model, X_test, y_test)

                                # Display the results
                                st.subheader("Model Performance")
                                st.write(f"Accuracy: {accuracy:.4f}")
                                st.write(f"ROC AUC Score: {roc_auc:.4f}")

                                # Save the model to session state
                                st.session_state.model = model
                                st.session_state.toxicity_categories = toxicity_columns

                                st.success("Model trained successfully!")

                                # Create a download link for the model
                                st.markdown(get_model_download_link(model, "toxic_comment_classifier.pkl"),
                                            unsafe_allow_html=True)

                                # Plot the toxicity distribution
                                st.subheader("Toxicity Distribution")
                                fig = plot_toxicity_distribution(df, toxicity_columns)
                                st.pyplot(fig)

                                # Display detailed metrics in debug mode
                                if st.session_state.debug_mode:
                                    st.subheader("Detailed Metrics")

                                    # Classification report
                                    st.write("Classification Report:")
                                    report = classification_report(y_test, predictions, target_names=toxicity_columns)
                                    st.text(report)

                                    # Confusion matrix for each category
                                    st.write("Confusion Matrix for Each Category:")
                                    for i, category in enumerate(toxicity_columns):
                                        st.write(f"Category: {category}")
                                        cm = pd.crosstab(y_test[category], predictions[:, i],
                                                         rownames=['Actual'], colnames=['Predicted'])
                                        st.write(cm)
            except Exception as e:
                st.error(f"Error processing training data: {str(e)}")
                if st.session_state.debug_mode:
                    import traceback
                    st.write("Traceback:", traceback.format_exc())
        else:
            st.info("Please upload a CSV file with labeled data to train a new model.")

    # Add a footer
    st.markdown("---")
    st.markdown("Toxic Comment Classifier and Moderator | Built with Streamlit")

# Call the main function when the script is run
if __name__ == "__main__":
    main()