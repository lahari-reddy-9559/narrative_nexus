import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation, NMF
import pandas as pd
import numpy as np
import streamlit as st
import random

# --- 0. NLTK Data Downloads (Required for the analysis functions) ---
# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# --- 1. CORE PREPROCESSING FUNCTION ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Tokenizes, removes punctuation/stopwords, and lemmatizes the input text."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# --- 2. MOCK MODEL TRAINING (Simulating a trained system) ---

# Define the number of topics for the models
N_TOPICS = 3

# Sample data for mock training (must be done to initialize the vectorizer and models)
MOCK_DOCUMENTS = [
    "The new strategy is highly effective and generates positive financial results.",
    "The quarterly report was entirely neutral, neither good nor bad, just facts and figures.",
    "Critical failure in production due to lack of support and a complex, broken methodology.",
    "Integration testing went flawlessly, exceeding all expectations. Fantastic job.",
    "The documentation is too complex and the project is running late, causing severe stress.",
    "New features are promising, but future scaling is a major concern. Needs review."
]
MOCK_LABELS = ['positive', 'neutral', 'negative', 'positive', 'negative', 'neutral']

# 1. Clean the mock training data
cleaned_documents = [clean_text(doc) for doc in MOCK_DOCUMENTS]

# 2. TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

# 3. Logistic Regression (Sentiment Model)
logreg = LogisticRegression()
logreg.fit(tfidf_matrix, MOCK_LABELS)

# 4. LDA (Topic Model 1)
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42)
lda.fit(tfidf_matrix)

# 5. NMF (Topic Model 2)
nmf = NMF(n_components=N_TOPICS, init='random', random_state=42, max_iter=500)
nmf.fit(tfidf_matrix)

# Helper function to generate topic names based on top words
def generate_topic_names(model, feature_names, no_top_words=3):
    """Generates a list of descriptive topic names."""
    topic_names = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topic_names.append(f"Topic {topic_idx+1}: {top_words}")
    return topic_names

lda_topic_names = generate_topic_names(lda, feature_names)
nmf_topic_names = generate_topic_names(nmf, feature_names)

# --- 3. ANALYSIS FUNCTION (User's provided logic) ---

def analyze_text(raw_text):
    """
    Analyzes the sentiment and topics of a given raw text using pre-trained models.
    """
    # Preprocess the input text
    cleaned_text = clean_text(raw_text)

    # Handle case where text becomes empty after cleaning
    if not cleaned_text:
        return 'neutral', {}, {}

    # Transform the cleaned text using the trained TF-IDF vectorizer
    user_tfidf_matrix = tfidf_vectorizer.transform([cleaned_text])

    # Predict sentiment using the trained Logistic Regression model
    predicted_sentiment = logreg.predict(user_tfidf_matrix)[0]

    # Get topic distributions using the trained LDA and NMF models
    user_lda_topic_distribution_array = lda.transform(user_tfidf_matrix)[0]
    user_nmf_topic_distribution_array = nmf.transform(user_tfidf_matrix)[0]

    # Create dictionaries for topic distributions with topic names
    lda_topic_distribution = dict(zip(lda_topic_names, user_lda_topic_distribution_array))
    nmf_topic_distribution = dict(zip(nmf_topic_names, user_nmf_topic_distribution_array))

    return predicted_sentiment, lda_topic_distribution, nmf_topic_distribution

# --- 4. STREAMLIT UI AND DISPLAY FUNCTIONS ---

# Set Streamlit page configuration for a wide, clean layout
st.set_page_config(
    page_title="NarrativeNexus Text Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the deep violet and cyan palette
st.markdown("""
<style>
    :root {
        --color-primary-bg: #1E1F39; /* Very Dark Indigo/Violet */
        --color-secondary-bg: #2C2D4A; /* Darker Indigo for panels */
        --color-accent: #4AF6F2; /* Bright Electric Cyan/Blue */
        --color-text: #E0E0E6; /* Near White */
        --color-negative: #FF7A7A; /* Soft Red */
        --color-neutral: #6C757D; /* Gray */
    }

    .stApp {
        background-color: var(--color-primary-bg);
        color: var(--color-text);
        font-family: 'Inter', sans-serif;
    }
    .stTextArea label, .stButton button, h1, h2, h3, .stMarkdown {
        color: var(--color-text) !important;
    }
    .stMarkdown h1 {
        color: var(--color-accent) !important;
    }

    /* Input/Textarea styling */
    .stTextArea textarea {
        background-color: #1a1a30;
        color: var(--color-text);
        border: 1px solid rgba(74, 246, 242, 0.4);
        border-radius: 0.5rem;
    }

    /* Button styling (Cyan accent) */
    .stButton button {
        background-color: var(--color-accent) !important;
        color: #1E1F39 !important;
        font-weight: bold;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background-color: #34c4c1 !important;
        box-shadow: 0 0 10px rgba(74, 246, 242, 0.7);
    }

    /* Metrics Box Styling */
    div[data-testid="stMetric"] > div:nth-child(1) {
        border-radius: 0.75rem;
        background-color: var(--color-secondary-bg);
        padding: 1rem;
        border: 1px solid rgba(74, 246, 242, 0.2);
    }
    div[data-testid="stMetricLabel"] p {
        color: #9ca3af; /* Gray text for label */
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def display_results(sentiment, lda_topics, nmf_topics):
    """Displays the analysis results in a dashboard format."""

    # 1. Sentiment Coloring
    if sentiment == 'positive':
        sentiment_color = '#4AF6F2' # Cyan
        sentiment_emoji = '✨'
    elif sentiment == 'negative':
        sentiment_color = '#FF7A7A' # Red
        sentiment_emoji = '🚨'
    else:
        sentiment_color = '#6C757D' # Gray
        sentiment_emoji = '⚪'

    # 2. Key Metrics
    st.markdown("---")
    st.subheader("Analysis Metrics")

    col1, col2, col3 = st.columns(3)

    # Metric 1: Sentiment
    with col1:
        st.metric(
            label="Predicted Sentiment",
            value=f"{sentiment_emoji} {sentiment.upper()}",
            delta_color="off"
        )
        st.markdown(f"""
        <style>
        div[data-testid="stMetricValue"] {{color: {sentiment_color};}}
        </style>
        """, unsafe_allow_html=True)

    # Metric 2: Tokens (Approximation)
    with col2:
        st.metric(
            label="Approx. Tokens",
            value=f"{len(st.session_state.text_input.split())}",
            delta_color="off"
        )
        st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {color: #E0E0E6;}
        </style>
        """, unsafe_allow_html=True)

    # Metric 3: Primary Theme (from LDA)
    primary_theme = max(lda_topics, key=lda_topics.get)
    primary_theme_score = lda_topics[primary_theme] * 100
    with col3:
        st.metric(
            label="Primary LDA Theme",
            value=f"{primary_theme.split(': ')[0]}",
            delta=f"{primary_theme_score:.1f}% Score",
            delta_color="off"
        )
        st.markdown(f"""
        <style>
        div[data-testid="stMetricValue"] {{color: #4AF6F2;}}
        div[data-testid="stMetricDelta"] {{color: #9ca3af;}}
        </style>
        """, unsafe_allow_html=True)

    # 3. Visualization Area (Topic Distributions)
    st.markdown("---")
    st.subheader("Topic Distribution Models")

    vis_col1, vis_col2 = st.columns(2)

    def prepare_topic_data(topic_dict):
        """Converts topic dict to DataFrame for plotting."""
        df = pd.DataFrame(topic_dict.items(), columns=['Topic', 'Score'])
        df['Topic_Name'] = df['Topic'].apply(lambda x: x.split(': ')[0])
        df['Score'] = df['Score'] * 100 # Convert to percentage
        return df.sort_values(by='Score', ascending=True)

    # Plot 1: LDA
    with vis_col1:
        st.markdown(f"### <span style='color:{sentiment_color}'>LDA Topic Prevalence</span>", unsafe_allow_html=True)
        lda_df = prepare_topic_data(lda_topics)
        st.bar_chart(lda_df, x='Topic_Name', y='Score', color="#4AF6F2")
        st.caption("LDA (Latent Dirichlet Allocation) focuses on modeling documents as a mixture of topics.")

    # Plot 2: NMF
    with vis_col2:
        st.markdown(f"### <span style='color:{sentiment_color}'>NMF Topic Prevalence</span>", unsafe_allow_html=True)
        nmf_df = prepare_topic_data(nmf_topics)
        # Use a slightly different color for distinction, but still cohesive
        st.bar_chart(nmf_df, x='Topic_Name', y='Score', color="#6873F6")
        st.caption("NMF (Non-negative Matrix Factorization) focuses on identifying parts-based representations.")


    # 4. Summary and Insights (Simulated advanced generation)
    st.markdown("---")
    st.subheader("Executive Summary & Insights (Simulated LLM Output)")

    # Placeholder for the summary based on the highest-scoring topic
    summary_text = (
        f"The system detected a predominant **{sentiment.upper()}** tone in the input text, strongly associated with the **{primary_theme.split(': ')[0]}** theme. "
        "The LDA and NMF models confirm distinct topic separation. This suggests the immediate priority is to address/leverage the content related to "
        f"*{primary_theme.split(': ')[1].split()[0]}* which drives the current sentiment profile."
    )

    with st.container(border=True):
        st.markdown(f"**Analysis Summary:** {summary_text}")

    st.markdown("---")

# Main application function
def main():
    st.title("Narrative<span style='color:#4AF6F2'>Nexus</span> Analysis Terminal")
    st.markdown(
        """
        Input your text data below. The system will preprocess the data, predict **sentiment** using
        Logistic Regression, and extract **themes** using LDA and NMF topic models.
        """
    )

    # User Input
    user_input = st.text_area(
        "Paste Data Stream Here (Max 5,000 Chars)",
        key="text_input",
        height=250,
        placeholder="E.g., The team's integration phase was a huge success, and the new feature is receiving positive reviews. This validates our future scaling efforts."
    )

    if st.button("🚀 Initiate Analysis Protocol"):
        if not user_input.strip():
            st.warning("Please enter some text to begin the analysis.")
            return

        with st.spinner('Tokenizing and running models... This may take a moment.'):
            # Simulate a 1-2 second processing delay
            time.sleep(random.uniform(1, 2))

            # Execute the analysis using the user's provided function
            try:
                sentiment, lda_topics, nmf_topics = analyze_text(user_input)

                # Display the results
                display_results(sentiment, lda_topics, nmf_topics)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)

    # Debug/Info for the models
    with st.expander("Model Configuration & Topic Mapping"):
        st.markdown(f"**Trained LDA Topics ({N_TOPICS}):**")
        st.code('\n'.join(lda_topic_names))
        st.markdown(f"**Trained NMF Topics ({N_TOPICS}):**")
        st.code('\n'.join(nmf_topic_names))
        st.caption("Note: Since models are trained on mock data, topic coherence may be low for arbitrary inputs.")


if __name__ == "__main__":
    main()