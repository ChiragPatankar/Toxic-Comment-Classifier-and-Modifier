# Toxic Comment Classifier and Moderator Streamlit App

This is a Streamlit web application that allows users to classify, moderate, and analyze text comments for toxicity and sentiment. It supports analyzing individual comments, processing comments in batches from a CSV file, and training a custom toxicity classification model using labeled data.

## Features

*   **Toxic Comment Classification:** Classifies text into multiple toxicity categories (Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate).
*   **Multi-Label Support:** Can classify a single comment into multiple categories simultaneously.
*   **Individual Comment Analysis:** Analyze a single comment and get detailed toxicity probabilities and sentiment.
*   **Batch Processing:** Upload a CSV file containing comments and get toxicity predictions, moderation suggestions, and sentiment analysis for all comments.
*   **Automatic Moderation:** Apply configurable thresholds to automatically mark comments for moderation (censoring) or deletion.
*   **Sentiment Analysis:** Provides sentiment polarity (Positive, Negative, Neutral) for comments.
*   **Model Training:** Train a new multi-label classification model using your own labeled dataset (CSV format).
*   **Model Persistence:** Load and save trained models using pickle files.
*   **Data Visualization:** Visualizes toxicity distributions, moderation actions, and sentiment distributions for batch processing results and training data.
*   **Data Validation:** Includes basic validation for uploaded training datasets.
*   **Downloadable Results:** Download the processed batch results as a CSV file.
*   **Downloadable Sample Data:** Provides a sample CSV format for training data.
*   **Debug Mode:** Toggle debug information in the sidebar for troubleshooting.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ChiragPatankar/Toxic-Comment-Classifier-and-Modifier.git
    cd Toxic-Comment-Classifier-and-Modifier
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:** Create a `requirements.txt` file in the same directory as your Python script (`app.py`) with the following content:

    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    nltk
    scikit-learn
    textblob
    ```

    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:** The script attempts to download 'stopwords' automatically. If you encounter NLTK data errors, you might need to run a Python interpreter once and execute the download manually:
    ```python
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    # You might also need 'punkt' for TextBlob depending on your environment
    # try:
    #     nltk.data.find('tokenizers/punkt')
    # except LookupError:
    #     nltk.download('punkt')
    ```

## Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py # Or the name of your Python script if different
    ```

2.  **Access the App:** The app will open in your web browser (usually `http://localhost:8501`).

3.  **Navigate:** Use the sidebar on the left to switch between pages:
    *   **Home:** Introduction and overview. Includes a button to show and download the sample data format for training/batch files.
    *   **Analyze Comments:** Input a single comment, get toxicity probabilities, sentiment, and moderation suggestions. You will need to either load a trained model here or train one on the "Train Model" page first.
    *   **Batch Processing:** Upload a CSV file containing comments (specify the comment column), process them all, and download the results. Requires a loaded/trained model.
    *   **Train Model:** Upload a labeled CSV dataset, select comment and label columns, choose a model type (Logistic Regression or Naive Bayes), split data, train, evaluate, and save the model.

## Data Format for Training and Batch Processing

### Training Data

For training a new model, your CSV file must contain:
*   A column with the text of the comments.
*   One or more columns representing toxicity categories, with binary values (0 or 1) indicating whether the comment belongs to that category (1) or not (0).

Example (`sample_toxic_data.csv`):

| comment_text                             | toxic | severe_toxic | obscene | threat | insult | identity_hate |
| :--------------------------------------- | :---- | :----------- | :------ | :----- | :----- | :------------ |
| This is a normal comment.                | 0     | 0            | 0       | 0      | 0      | 0             |
| This is a toxic comment you idiot!       | 1     | 0            | 1       | 0      | 1      | 0             |
| You're all worthless and should die.     | 1     | 1            | 0       | 1      | 1      | 0             |
| I respectfully disagree with your point. | 0     | 0            | 0       | 0      | 0      | 0             |

A sample CSV with this format is available for download on the Home page.

### Batch Processing Data

For batch processing, your CSV file primarily needs:
*   A column with the text of the comments that you want to analyze.

While the sample data shown might include label columns, the batch processing function *only requires* the comment column you select. The app will add new columns for toxicity probabilities, moderation action, and sentiment.

## Technical Details

*   **Framework:** Streamlit
*   **Data Handling:** Pandas
*   **NLP:** NLTK (Stopwords, Stemming), Regex, TextBlob (Sentiment)
*   **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`, `OneVsRestClassifier`, `train_test_split`, evaluation metrics)
*   **Visualization:** Matplotlib, Seaborn
*   **Model Serialization:** Pickle

The application utilizes a `Pipeline` with TF-IDF vectorization followed by a `OneVsRestClassifier` wrapping either `LogisticRegression` or `MultinomialNB` for multi-label classification.

## Screenshots

*(Optional: Add screenshots here to show how the app looks)*

## Contributing

Feel free to fork the repository and submit pull requests. Issues and feature requests are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with Streamlit*
