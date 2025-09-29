import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
import nltk
import os
import kagglehub

# NLTK Imports
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# FIX for ImportError: Dynamically access DownloadError to ensure compatibility
try:
    from nltk.downloader import DownloadError
except ImportError:
    # Fallback for environments (like some Streamlit Cloud setups) where 
    # the direct submodule import fails, but nltk.downloader is available.
    DownloadError = getattr(nltk.downloader, 'DownloadError', Exception)


# Scikit-learn Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Transformers Import
from transformers import pipeline

# --- Initial Setup and Global Variables ---

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Dynamic Text Analysis Platform")

# --- NLTK Data Download Function (FIXED) ---

def download_nltk_data():
    """Download necessary NLTK data."""
    # Use the robustly imported DownloadError to handle exceptions
    try:
        nltk.data.find('corpora/stopwords')
    except DownloadError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except DownloadError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except DownloadError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except DownloadError:
        nltk.download('punkt', quiet=True)

# Run NLTK download once at startup
download_nltk_data()

# Define preprocessing components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

@st.cache_data
def clean_text(text):
    """Preprocesses text: lowercases, removes punctuation, stopwords, and lemmatizes."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 1]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Define topic names
lda_topic_names = [
    "Daily Greetings and Wishes", "Social Interaction", "Low Urgency/Feedback", "Call to Action",
    "High Urgency/Crisis:", "Work/Professional Life:", "Personal Life/Relationships:",
    "Parenting/Childcare:", "General Commentary/Venting:", "Travel/Events:"
]
nmf_topic_names = [
    "Low Urgency/Feedback", "Call to Action", "High Urgency/Crisis:", "Work/Professional Life:",
    "Personal Life/Relationships:", "Parenting/Childcare:", "General Commentary/Venting:",
    "Travel/Events:", "Product/Service Mention:", "Daily Greetings and Wishes"
]

# --- Dataset Loading, Preprocessing, and Model Training (Cached) ---

@st.cache_resource(show_spinner="Loading Data and Training Models...")
def load_data_and_train_models():
    """Loads data, preprocesses it, and trains the necessary models."""
    df = pd.DataFrame()
    tfidf_vectorizer = None
    logreg = None
    lda = None
    nmf = None
    tfidf_matrix = None
    lda_matrix = None
    nmf_matrix = None
    summarization_pipeline = None

    try:
        # Load the dataset
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        file_path = os.path.join(path, 'train.csv')
        df = pd.read_csv(file_path, encoding='latin-1')

        # Handle missing values and sample for speed (important for Streamlit Cloud)
        df.dropna(subset=['text', 'selected_text', 'sentiment'], inplace=True)
        # Sample the data to ensure fast app loading
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)

        # Apply cleaning to the text column
        df['cleaned_text'] = df['text'].apply(clean_text)

        # Initialize Models
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        lda = LatentDirichletAllocation(n_components=10, max_iter=10, random_state=42)
        nmf = NMF(n_components=10, random_state=42, init='nndsvda', max_iter=200)

        # Fit TF-IDF Vectorizer
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

        # Train Logistic Regression model (Sentiment Classification)
        logreg.fit(tfidf_matrix, df['sentiment'])

        # Train LDA and NMF models (Topic Modeling)
        lda_matrix = lda.fit_transform(tfidf_matrix)
        nmf_matrix = nmf.fit_transform(tfidf_matrix)

        # Load Summarization Pipeline
        summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

    except Exception as e:
        st.error(f"Error during data loading or model training: {e}")
        st.warning("Analysis functions will be disabled due to model/data loading failure.")

    return df, tfidf_vectorizer, logreg, lda, nmf, tfidf_matrix, lda_matrix, nmf_matrix, summarization_pipeline

# Load all resources
df, tfidf_vectorizer, logreg, lda, nmf, tfidf_matrix, lda_matrix, nmf_matrix, summarization_pipeline = load_data_and_train_models()


# --- Analysis and Summarization Functions ---

def analyze_text(raw_text):
    """Analyzes the sentiment and topics of a given raw text."""
    if df.empty or tfidf_vectorizer is None or logreg is None or lda is None or nmf is None:
        return "N/A (Models not loaded)", None, None

    cleaned_text = clean_text(raw_text)
    user_tfidf_matrix = tfidf_vectorizer.transform([cleaned_text])

    # Predict sentiment
    predicted_sentiment = logreg.predict(user_tfidf_matrix)[0]

    # Get topic distributions
    user_lda_topic_distribution_array = lda.transform(user_tfidf_matrix)[0]
    user_nmf_topic_distribution_array = nmf.transform(user_tfidf_matrix)[0]

    # Create dictionaries for topic distributions
    lda_topic_distribution = dict(zip(lda_topic_names, user_lda_topic_distribution_array))
    nmf_topic_distribution = dict(zip(nmf_topic_names, user_nmf_topic_distribution_array))

    return predicted_sentiment, lda_topic_distribution, nmf_topic_distribution

def summarize_texts_by_topic_sentiment(topic_name, sentiment, topic_threshold=0.2, max_summaries=5):
    """Filters data and generates summaries."""
    if df.empty or lda_matrix is None or nmf_matrix is None or summarization_pipeline is None:
        return []

    # Prepare combined DataFrame for filtering
    temp_df = df.copy()
    lda_df_temp = pd.DataFrame(lda_matrix, columns=lda_topic_names)
    
    # Rename NMF columns for unique identification
    nmf_cols_safe = [f"{col}_NMF" for col in nmf_topic_names]
    nmf_df_temp = pd.DataFrame(nmf_matrix, columns=nmf_cols_safe)

    # Re-align indexes before concatenation
    temp_df = temp_df.reset_index(drop=True)
    lda_df_temp = lda_df_temp.reset_index(drop=True)
    nmf_df_temp = nmf_df_temp.reset_index(drop=True)
    combined_df = pd.concat([temp_df, lda_df_temp, nmf_df_temp], axis=1)

    # Determine which topic distribution to use
    filter_col = None
    if topic_name in lda_topic_names:
        filter_col = topic_name
    elif topic_name in nmf_topic_names:
        filter_col = f"{topic_name}_NMF"

    if not filter_col:
        return []

    # Filter the DataFrame
    filtered_df = combined_df[
        (combined_df[filter_col] > topic_threshold) &
        (combined_df['sentiment'] == sentiment)
    ]

    if filtered_df.empty:
        return []

    # Select a subset of texts for summarization
    texts_to_summarize = filtered_df['text'].head(max_summaries).tolist()

    # Generate summaries
    summaries = summarization_pipeline(texts_to_summarize, max_length=150, min_length=30, do_sample=False)

    # Format the results
    summarized_results = []
    for i, summary_info in enumerate(summaries):
        summarized_results.append({
            "original_text": texts_to_summarize[i],
            "summary": summary_info['summary_text']
        })

    return summarized_results

# --- Visualization and Recommendation Functions ---

def get_sentiment_distribution_df():
    """Calculates the mean topic distribution by sentiment for heatmaps."""
    if df.empty or lda_matrix is None or nmf_matrix is None:
        return None, None

    # Prepare combined DataFrame for calculations
    lda_df_temp = pd.DataFrame(lda_matrix, columns=lda_topic_names)
    
    # Rename NMF columns for unique identification and then rename back for display
    nmf_df_temp = pd.DataFrame(nmf_matrix, columns=[f"{col}_NMF" for col in nmf_topic_names])

    combined_df = pd.concat([df.reset_index(drop=True), lda_df_temp, nmf_df_temp], axis=1)

    lda_sentiment_distribution = combined_df.groupby('sentiment')[lda_topic_names].mean()
    
    # Recalculate NMF sentiment distribution and rename columns for display
    nmf_sentiment_distribution = combined_df.groupby('sentiment')[nmf_df_temp.columns].mean()
    nmf_sentiment_distribution.columns = nmf_topic_names

    return lda_sentiment_distribution, nmf_sentiment_distribution


def plot_sentiment_distribution(sentiment_series):
    """Plots the overall sentiment distribution."""
    if sentiment_series is None or sentiment_series.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=sentiment_series, palette='viridis', ax=ax)
    ax.set_title('Overall Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    return fig

def plot_topic_sentiment_heatmap(sentiment_distribution_df, title):
    """Plots a heatmap of mean topic distribution by sentiment."""
    if sentiment_distribution_df is None or sentiment_distribution_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(sentiment_distribution_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Topics')
    ax.set_ylabel('Sentiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def generate_wordcloud(topic_components, feature_names, topic_name, no_top_words=50):
    """Generates a word cloud for a given topic."""
    if topic_components is None or feature_names is None:
        return None
    # Use features names from the vectorizer
    top_words_dict = {feature_names[i]: topic_components[i] for i in topic_components.argsort()[:-no_top_words - 1:-1]}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f"Topic: {topic_name}")
    return fig

def get_actionable_recommendation(sentiment, lda_topics, nmf_topics):
    """Provides actionable insights based on analysis."""
    recommendation = ""
    # Use LDA for primary topic check for simplicity in demonstration
    high_urgency_topic = lda_topics.get("High Urgency/Crisis:", 0)
    product_mention_topic = lda_topics.get("Product/Service Mention:", 0)

    if sentiment == 'negative':
        recommendation += "🚨 **Immediate Action Recommended:** Negative sentiment detected. "
        if high_urgency_topic > 0.3:
            recommendation += "High Urgency/Crisis topic is dominant. **Prioritize immediate investigation** and response to mitigate potential damage. Use the summarization tool to extract key complaints. "
        elif product_mention_topic > 0.3:
            recommendation += "Focus on **specific product/service complaints**. Initiate a deeper root cause analysis on this area. "
        else:
            recommendation += "Consider a **deeper dive into 'General Commentary/Venting'** to identify underlying issues before they escalate. "

    elif sentiment == 'positive':
        recommendation += "✅ **Strategy Recommendation:** Positive sentiment detected. "
        if lda_topics.get("Social Interaction", 0) > 0.3:
            recommendation += "Identify and **amplify positive social interactions** as testimonials or success stories. "
        else:
            recommendation += "**Benchmark successful processes** associated with the highest contributing topic to replicate success. "

    elif sentiment == 'neutral':
        recommendation += "💡 **Investigation Suggestion:** Neutral sentiment detected. "
        recommendation += "Investigate the **'Low Urgency/Feedback'** topic for potential minor improvements or overlooked suggestions that could boost satisfaction. "

    if not recommendation:
        recommendation = "No specific recommendation generated. Ensure your text is relevant to the trained topics."

    return recommendation

# ==============================================================================
# --- Streamlit Application Layout ---
# ==============================================================================

st.title("🚀 Dynamic Text Analysis Platform")
st.markdown("""
<div style='text-align: justify; padding: 10px; border-radius: 5px; background-color: #f0f2f6;'>
The **Dynamic Text Analysis Platform** processes diverse text inputs to extract key themes, sentiment, and actionable insights. It leverages advanced NLP (TF-IDF, Logistic Regression for Sentiment, LDA/NMF for Topic Modeling, and BART for Summarization) to transform raw data into concise, decision-making outputs.
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Platform Navigation")
tab1, tab2, tab3 = st.tabs(["Text Analyzer & Insights", "Summarization Tool", "Data Visualizations"])


# --- TAB 1: Text Analyzer & Insights ---
with tab1:
    st.header("Analyze New Text & Get Actionable Insights")
    st.markdown("Enter any article, report, or social media content to instantly predict its **Sentiment**, identify **Key Topics**, and receive **Actionable Recommendations**.")

    user_text_input = st.text_area("✍️ Enter text here (min 50 words recommended):", height=200, key="analyze_text_input")

    if st.button("Analyze Text & Recommend Action", use_container_width=True):
        if user_text_input and len(user_text_input.split()) > 10:
            with st.spinner("Processing analysis..."):
                sentiment, lda_topics, nmf_topics = analyze_text(user_text_input)

            if sentiment != "N/A (Models not loaded)":
                st.subheader("Analysis Results")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="Predicted Sentiment", value=sentiment.upper())

                # --- Actionable Insights/Recommendation Engine ---
                with st.expander("⭐ Actionable Recommendation (Insight Engine)", expanded=True):
                    recommendation = get_actionable_recommendation(sentiment, lda_topics if lda_topics else {}, nmf_topics if nmf_topics else {})
                    st.info(recommendation)

                st.markdown("---")

                # Topic Distribution Display
                col4, col5 = st.columns(2)
                with col4:
                    st.subheader("LDA Topic Distribution")
                    if lda_topics:
                        lda_df = pd.DataFrame(lda_topics.items(), columns=['Topic', 'Contribution'])
                        lda_df = lda_df.sort_values(by='Contribution', ascending=False)
                        st.dataframe(lda_df.head(5).style.format({"Contribution": "{:.2%}"}), use_container_width=True, hide_index=True)
                with col5:
                    st.subheader("NMF Topic Distribution")
                    if nmf_topics:
                        nmf_df = pd.DataFrame(nmf_topics.items(), columns=['Topic', 'Contribution'])
                        nmf_df = nmf_df.sort_values(by='Contribution', ascending=False)
                        st.dataframe(nmf_df.head(5).style.format({"Contribution": "{:.2%}"}), use_container_width=True, hide_index=True)
            else:
                st.error("Cannot perform analysis. Please check the console for data/model loading errors.")
        else:
            st.warning("Please enter a meaningful amount of text (more than 10 words) to analyze.")

# --- TAB 2: Summarization Tool ---
with tab2:
    st.header("Batch Summarization by Topic & Sentiment")
    st.markdown("Generate concise summaries of texts from the dataset that match a specific **Topic** and **Sentiment** criteria.")

    if summarization_pipeline is None or df.empty:
        st.warning("Summarization is disabled due to missing data or pipeline failure.")
    else:
        # Create a combined list of unique topic names for the select box
        all_topics = list(set(lda_topic_names + nmf_topic_names))
        
        col_select, col_slider = st.columns([1, 1])

        with col_select:
            summary_topic = st.selectbox("🎯 Select Topic for Summarization:", all_topics, key="summary_topic")
            summary_sentiment = st.selectbox("😊 Select Target Sentiment:", ['negative', 'neutral', 'positive'], key="summary_sentiment")

        with col_slider:
            summary_threshold = st.slider("⚖️ Min Topic Contribution Threshold:", 0.0, 0.5, 0.2, 0.05, key="summary_threshold", help="Only documents where the selected topic contributes more than this percentage will be included.")
            num_summaries = st.slider("📝 Max Number of Summaries to Generate:", 1, 10, 3, key="num_summaries")

        if st.button("Generate Summaries", use_container_width=True):
            with st.spinner(f"Filtering and generating {num_summaries} summaries..."):
                summaries_list = summarize_texts_by_topic_sentiment(summary_topic, summary_sentiment, summary_threshold, num_summaries)

            if summaries_list:
                st.subheader(f"Summaries for **'{summary_topic}'** with **'{summary_sentiment.upper()}'** sentiment:")
                for i, item in enumerate(summaries_list):
                    st.markdown(f"**{i+1}. Original Text:** *{item['original_text']}*")
                    st.success(f"**Summary:** {item['summary']}")
                    st.markdown("---")
            else:
                st.info(f"No documents found matching Topic: **{summary_topic}**, Sentiment: **{summary_sentiment}**, and Threshold: > {summary_threshold:.2f}.")

# --- TAB 3: Data Visualizations ---
with tab3:
    st.header("Data Visualizations & Exploratory Analytics")
    st.markdown("Explore the underlying distribution of sentiment and the relationship between topics and sentiment.")

    if df.empty or lda_matrix is None or nmf_matrix is None:
        st.warning("Data and/or models are not available for visualizations.")
    else:
        # 1. Overall Sentiment Distribution
        st.subheader("1. Overall Sentiment Distribution")
        sentiment_fig = plot_sentiment_distribution(df['sentiment'])
        if sentiment_fig:
            st.pyplot(sentiment_fig)
        st.markdown("---")

        lda_dist, nmf_dist = get_sentiment_distribution_df()

        # 2. Topic-Sentiment Heatmaps
        st.subheader("2. Topic-Sentiment Relationship (Mean Topic Contribution by Sentiment)")
        col_lda, col_nmf = st.columns(2)

        with col_lda:
            st.markdown("#### LDA Topic Distribution")
            lda_heatmap = plot_topic_sentiment_heatmap(lda_dist, 'Mean LDA Topic Distribution by Sentiment')
            if lda_heatmap:
                st.pyplot(lda_heatmap)

        with col_nmf:
            st.markdown("#### NMF Topic Distribution")
            nmf_heatmap = plot_topic_sentiment_heatmap(nmf_dist, 'Mean NMF Topic Distribution by Sentiment')
            if nmf_heatmap:
                st.pyplot(nmf_heatmap)
        st.markdown("---")

        # 3. Topic Word Clouds
        st.subheader("3. Topic Word Clouds (Key Terms for Topics)")
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

        wordcloud_topic_type = st.radio("Select Topic Model:", ['LDA', 'NMF'], key="wc_model_select")

        if wordcloud_topic_type == 'LDA':
            wc_topic_name = st.selectbox("Select LDA Topic to Visualize:", lda_topic_names, key="wc_lda_select")
            topic_idx = lda_topic_names.index(wc_topic_name)
            wc_fig = generate_wordcloud(lda.components_[topic_idx], tfidf_feature_names, wc_topic_name)
            if wc_fig:
                st.pyplot(wc_fig)
        else: # NMF
            wc_topic_name = st.selectbox("Select NMF Topic to Visualize:", nmf_topic_names, key="wc_nmf_select")
            topic_idx = nmf_topic_names.index(wc_topic_name)
            wc_fig = generate_wordcloud(nmf.components_[topic_idx], tfidf_feature_names, wc_topic_name)
            if wc_fig:
                st.pyplot(wc_fig)
